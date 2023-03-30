import numpy as np
import tqdm
import torch
from sklearn.linear_model import SGDClassifier
import time
from torch.optim import SGD, Adam
import random
import sklearn
import wandb
import plac

EVAL_CLF_PARAMS = {"loss": "log", "tol": 1e-4, "iters_no_change": 15, "alpha": 1e-4, "max_iter": 25000}
NUM_CLFS_IN_EVAL = 3 # change to 1 for large dataset / high dimensionality

def init_classifier():

    return SGDClassifier(loss=EVAL_CLF_PARAMS["loss"], fit_intercept=True, max_iter=EVAL_CLF_PARAMS["max_iter"], tol=EVAL_CLF_PARAMS["tol"], n_iter_no_change=EVAL_CLF_PARAMS["iters_no_change"],
                        n_jobs=32, alpha=EVAL_CLF_PARAMS["alpha"])
                        
def symmetric(X):
    X.data = 0.5 * (X.data + X.data.T)
    return X

def get_score(X_train, y_train, X_dev, y_dev, P, rank):
    P_svd = get_projection(P, rank)
    
    loss_vals = []
    accs = []
    
    for i in range(NUM_CLFS_IN_EVAL): # See how well a fresh linear classifier could learn the projected data?
        clf = init_classifier()
        clf.fit(X_train@P_svd, y_train)
        y_pred = clf.predict_proba(X_dev@P_svd)
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        accs.append(clf.score(X_dev@P_svd, y_dev))
        
    i = np.argmin(loss_vals)
    return loss_vals[i], accs[i]


# In order to do the projection onto the convex hull, you need to solve some linear constraints regarding the eigenvalues. This does that solving
def solve_constraint(lambdas, d=1): # note: lambdas has shape (hidden_sz,) and is the "D" matrix (whatever that is). d is the rank (number of dimensions) of the subspace we want to use to represent the subspace
    def f(theta): # TODO: what's this function do? What's theta? 
        return_val = np.sum(np.minimum(np.maximum(lambdas - theta, 0), 1)) - d
        return return_val

    theta_min, theta_max = max(lambdas), min(lambdas) - 1 # TODO: What do the two thetas represent?
    assert f(theta_min) * f(theta_max) < 0 # TODO: why do we need this assertion?

    mid = (theta_min + theta_max) / 2
    tol = 1e-4
    iters = 0

    while iters < 25: # *** TODO: what is this? some sort of binary search? What is it trying to converge to? ***

        mid = (theta_min + theta_max) / 2

        if f(mid) * f(theta_min) > 0:

            theta_min = mid
        else:
            theta_max = mid
        iters += 1

    lambdas_plus = np.minimum(np.maximum(lambdas - mid, 0), 1) # shape: (hidden_sz=200,) TODO: why is matrix mostly 0s excetp the last 7 (out of 200) values?
    # if (theta_min-theta_max)**2 > tol:
    #    print("didn't converge", (theta_min-theta_max)**2)
    return lambdas_plus

def get_majority_acc(y):

    from collections import Counter
    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    maj = max(fracts)
    return maj

def get_entropy(y):

    from collections import Counter
    import scipy
    
    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    return scipy.stats.entropy(fracts)
    

def get_projection(P, rank): # How is this getting the projection - literally just SVD (0 out all the eigenvalues that you want to not be 1)
    # This is the final SVD to convert a P on the convex hull to a actual valid projection matrix
    D,U = np.linalg.eigh(P)
    U = U.T
    W = U[-rank:]
    P_final = np.eye(P.shape[0]) - W.T @ W # this is the P = I − W.T * W, I guess
    return P_final

def prepare_output(P,rank,score):
    P_final = get_projection(P,rank)
    return {"score": score, "P_before_svd": np.eye(P.shape[0]) - P, "P": P_final}

def solve_adv_game(X_train, y_train, X_dev, y_dev, rank=1, device="cpu", out_iters=75000, in_iters_adv=1, in_iters_clf=1, epsilon=0.0015, batch_size=128, evalaute_every=1000, optimizer_class=SGD, 
optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4}, optimizer_params_predictor={"lr": 0.005, "weight_decay": 1e-4}):
    """

    :param X: The input (np array)
    :param Y: the lables (np array)
    :param X_dev: Dev set (np array)
    :param Y_dev: Dev labels (np array)
    :param rank: Number of dimensions to neutralize from the input.
    :param device:
    :param out_iters: Number of batches to run
    :param in_iters_adv: number of iterations for adversary's optimization
    :param in_iters_clf: number of iterations from the predictor's optimization
    :param epsilon: stopping criterion .Stops if abs(acc - majority) < epsilon.
    :param batch_size:
    :param evalaute_every: After how many batches to evaluate the current adversary.
    :param optimizer_class: SGD/Adam etc.
    :param optimizer_params: the optimizer's params (as a dict)
    :return:
    """

    def get_loss_fn(X, y, predictor, P, bce_loss_fn, optimize_P=False):
        I = torch.eye(X_train.shape[1]).to(device) # I.shape = (d, d)
        
        # P = the projection matrix onto the concept's subspace 
        #  (aka X @ P would project X to just the subspace identifying the concept and 
        #   throw out all other info; if rank=1, visually imagine this is projecting 
        #   from high dim space to either side of a line)
        # (I - P) = the projection matrix onto the orthogonal complement of the concept subspace 
        #  (aka ideally, X @ (I - P) removes ALL ability to predict y)
        # (I - P).shape == P.shape == (d, d), where d=hidden state size.
        # X.shape = (batch_size, d)
        # (X @ (I - P)).shape = (batch_size, d) -
        #  basically, project from d dimensional space to a new d-dimensional space, 
        #  but the new d-dimensional space is orthogonal to P
        # Not sure why we need the squeeze, didn't seem to change the shape
        # What's the relationship between X and (X @ (I-P))?
        bce = bce_loss_fn(predictor(X @ (I - P)).squeeze(), y) 
        if optimize_P: # optimizing P means you want to maximize the loss (make the predictor as bad as possible at predicting the label y), so flip the sign
            bce = -bce
        return bce


    X_torch = torch.tensor(X_train).float().to(device)
    y_torch = torch.tensor(y_train).float().to(device)

    num_labels = len(set(y_train.tolist()))
    if num_labels == 2:
        predictor = torch.nn.Linear(X_train.shape[1], 1).to(device)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        y_torch = y_torch.float()
    else:
        # TODO: does this work for multiclass "concepts" too?
        # Yes, the difference would be to use num_labels outputs instead of just 1 for the predictive logit
        # Shauli's intuition: should work as well, but hasn't been empirically attempted
        # Since softmax is part of the CEL, this nonlinearity might be enough to overcome the linear erasure
        predictor = torch.nn.Linear(X_train.shape[1], num_labels).to(device)
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        y_torch = y_torch.long()

    P = 1e-1*torch.randn(X_train.shape[1], X_train.shape[1]).to(device) # shape: (d, d), where d is the dimensionality of the hidden state
    P.requires_grad = True

    optimizer_predictor = optimizer_class(predictor.parameters(), **optimizer_params_predictor)
    optimizer_P = optimizer_class([P],**optimizer_params_P)

    maj = get_majority_acc(y_train)
    label_entropy = get_entropy(y_train)
    pbar = tqdm.tqdm(range(out_iters), total = out_iters, ascii=True)
    count_examples = 0
    best_P, best_score, best_loss = None, 1, -1

    for i in pbar:

        for j in range(in_iters_adv):
            P = symmetric(P) # TODO: Enforce that P is symmetric? Why does this need to happen?
            # Goal: optimize over set of all projection matrices, but this is not convex. So, we need to relax this.
            # We start with an arbitrary matrix P which not necessarily a valid projection matrix.
            # Some constraints of valid projection matrices: must be symmetric, SVD-thing (you might have a matrix that's "close" to a projection matrix but is not actually a projection matrix. Then, there's a formula that uses SVD to project from a "close-to-projection matrix" to an actual valid projection matrix. Teeeechnically it's not guaranteed to be a valid projection matrix, but it will be very close bc it's on the convex hull. Then 1 more SVD to bring it from very close to actually valid)
            optimizer_P.zero_grad()

            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]] # randomly select batch_size number of train points + labels

            loss_P = get_loss_fn(X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=True)
            loss_P.backward()
            optimizer_P.step() # slightly move P and other things toward the goal

            # project

            with torch.no_grad():
                D, U = torch.linalg.eigh(symmetric(P).detach().cpu()) # TODO: what's going on here? # D.shape = (d,), U.shape = (d, d)
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D, d=rank)
                D = torch.tensor(np.diag(D_plus_diag).real).float().to(device) # TODO: what's this doing?
                U = U.to(device)
                P.data = U @ D @ U.T # TODO: help what is this, how is this updating P? ans: final step in projecting to convex hull of the set of valid projection matrices

        for j in range(in_iters_clf): # this makes sense, this is the part where the linear layer tries to minimize the loss
            optimizer_predictor.zero_grad()
            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

            loss_predictor = get_loss_fn(X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=False)
            loss_predictor.backward()
            optimizer_predictor.step()
            count_examples += batch_size

        if i % evalaute_every == 0:
            #pbar.set_description("Evaluating current adversary...")
            loss_val, score = get_score(X_train, y_train, X_train, y_train, P.detach().cpu().numpy(), rank)
            if loss_val > best_loss:#if np.abs(score - maj) < np.abs(best_score - maj):
                best_P, best_loss = symmetric(P).detach().cpu().numpy().copy(), loss_val
            if np.abs(score - maj) < np.abs(best_score - maj):
                best_score = score
                
            # update progress bar
            
            best_so_far = best_score if np.abs(best_score-maj) < np.abs(score-maj) else score
            
            # post-projection = current accuracy of a linear probe trained on the projected embeddings
            # best so-far = accuracy of the best run (aka closest accuracy to the Maj of around 50%)
            # Maj = the % of the data in the majority class, basically our baseline performance for a majority-class guesser. This should be our accuracy if the concept has been completely removed!
            # Gap = best so-far - Maj accuracy, indicates how close we are to completely erasing the performance of linear probe on this concept.
            
            wandb.log({
                "train":{
                    "iters": i,
                    "accuracy_on_projected": score,
                    "accuracy_gap": np.abs(best_so_far - maj),
                    "loss_val": loss_val,
                }
            })
            pbar.set_description("{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%; Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}".format(i, out_iters, score * 100, best_so_far * 100, maj * 100, np.abs(best_so_far - maj) * 100, best_loss, loss_val))
            pbar.refresh()  # to show immediately the update
            time.sleep(0.01)

        if i > 1 and np.abs(best_score - maj) < epsilon:
        #if i > 1 and np.abs(best_loss - label_entropy) < epsilon:
                    break

    # TODO: is this supposed to take like 10 min? depends on the dimensionality of vector - ideally keep it between 100-300. >700 should still work but will take longer
    output = prepare_output(best_P,rank,best_score)
    return output


@plac.opt('NUM_CLASSES', "Number of classes of the concept we'd like to erase", type=int)
@plac.opt('RANK', "Rank of the subspace we will try to represent the concept in", type=int)
@plac.opt('SEED', "Random seed", type=int)
def main(NUM_CLASSES=2, RANK=1, SEED=0):
    print(f"Running with parameters: {locals()}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # create a synthetic dataset
    n, dim = 15000, 200
    num_classes = NUM_CLASSES

    # arguments
    num_iters = 50000
    rank = RANK
    optimizer_class = torch.optim.SGD
    optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
    optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
    epsilon = 0.001 # stop 0.1% from majority acc
    batch_size = 256

    params_to_log = locals()
    PROJECT_NAME = "rlace_rank_vs_classes"
    TAGS = []
    run = wandb.init(
        project=PROJECT_NAME,
        # group=GROUP_NAME,
        config=params_to_log,
        tags=TAGS,
    )

    # randomly generate data and labels
    X = np.random.randn(n, dim)
    y = np.random.randint(low = 0, high = num_classes, size = n) #(np.random.rand(n) > 0.5).astype(int)

    # Now add some signal of the label into the data
    X[:, 0] = (y + np.random.randn(*y.shape) * 0.3) ** 2 + 0.3 * y
    X[:, 1] = (y + np.random.randn(*y.shape) * 0.1) ** 2 - 0.7 * y
    X[:, 2] = (y + np.random.randn(*y.shape) * 0.3) ** 2 + 0.5 * y + np.random.randn(*y.shape) * 0.2
    X[:, 3] = (y + np.random.randn(*y.shape) * 0.5) ** 2 - 0.7 * y + np.random.randn(*y.shape) * 0.1
    X[:, 4] = (y + np.random.randn(*y.shape) * 0.5) ** 2 - 0.8 * y + np.random.randn(*y.shape) * 0.1
    X[:, 5] = (y + np.random.randn(*y.shape) * 0.25) ** 2 - 0.2 * y + np.random.randn(*y.shape) * 0.1
    mixing_matrix = 1e-2*np.random.randn(dim, dim)
    X = X @ mixing_matrix
    
    l_train = int(0.6*n)
    X_train, y_train = X[:l_train], y[:l_train]
    X_dev, y_dev = X[l_train:], y[l_train:]

    output = solve_adv_game(X_train, y_train, X_dev, y_dev, rank=rank, device="cpu", out_iters=num_iters, optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon,batch_size=batch_size)
    wandb.log(output)
    # train a classifier
    
    P_svd = output["P"]
    P_before_svd = output["P_before_svd"]
    svm = init_classifier()
                        
    svm.fit(X_train[:] , y_train[:])
    score_original = svm.score(X_dev, y_dev)
    
    svm = init_classifier()
    svm.fit(X_train[:] @ P_before_svd , y_train[:])
    score_projected_no_svd = svm.score(X_dev @ P_before_svd, y_dev)
    
    svm = init_classifier()
    svm.fit(X_train[:] @ P_svd , y_train[:])
    score_projected_svd_dev = svm.score(X_dev @ P_svd, y_dev)
    score_projected_svd_train = svm.score(X_train @ P_svd, y_train)
    maj_acc_dev = get_majority_acc(y_dev)
    maj_acc_train = get_majority_acc(y_train)

    wandb.log({
        "test": {
            "accuracy_pre_projection": score_original,
            "accuracy_projected_no_svd": score_projected_no_svd,
            "accuracy_projected_svd": score_projected_svd_dev,
            "gap_accuracy_projected_no_svd": np.abs(maj_acc_dev - score_projected_no_svd),
            "gap_accuracy_projected_svd": np.abs(maj_acc_dev - score_projected_svd_dev),
        }
    })
    
    print("===================================================")
    print("Original Acc, dev: {:.3f}%; Acc, projected, no svd, dev: {:.3f}%; Acc, projected+SVD, train: {:.3f}%; Acc, projected+SVD, dev: {:.3f}%".format(
        score_original*100, score_projected_no_svd*100, score_projected_svd_train*100, score_projected_svd_dev*100))    
    print("Majority Acc, dev: {:.3f} %".format(maj_acc_dev*100))
    print("Majority Acc, train: {:.3f} %".format(maj_acc_train*100))
    print("Gap, dev: {:.3f} %".format(np.abs(maj_acc_dev - score_projected_svd_dev)*100))
    print("Gap, train: {:.3f} %".format(np.abs(maj_acc_train - score_projected_svd_train)*100))
    print("===================================================")
    eigs_before_svd, _ = np.linalg.eigh(P_before_svd)
    print("Eigenvalues, before SVD: {}".format(eigs_before_svd[:]))
    
    eigs_after_svd, _ = np.linalg.eigh(P_svd)
    print("Eigenvalues, after SVD: {}".format(eigs_after_svd[:]))

    # this is used to check whether you need a final SVD to convert the P on the convex hull to an actual projection matrix.
    
    eps = 1e-6
    assert np.abs( (eigs_after_svd > eps).sum() -  (dim - rank) ) < eps
    
    # TODO: what's the significance of these eigenvalues? --> ans: In this case, since the eigenvalues are all basically 1, this is essentially a valid projection matrix and no further SVD needed.
    # Note: a projection matrix with shape (d, d) of rank k should have k eigenvalues that = 1 and d - k eigenvalues = 0. Since this is the projection matrix onto the orthogonal complement of the rank-1 concept subspace, we expect 1 eigenvalue=0 and d-1=1.
    """
    Eigenvalues, before SVD: [-4.23114083e-07  9.99999993e-01  9.99999994e-01  9.99999995e-01
    9.99999995e-01  9.99999996e-01  9.99999996e-01  9.99999996e-01
    9.99999997e-01  9.99999997e-01  9.99999997e-01  9.99999997e-01
    9.99999997e-01  9.99999997e-01  9.99999997e-01  9.99999997e-01
    9.99999998e-01  9.99999998e-01  9.99999998e-01  9.99999998e-01
    9.99999998e-01  9.99999998e-01  9.99999998e-01  9.99999998e-01
    9.99999998e-01  9.99999998e-01  9.99999998e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000001e+00  1.00000001e+00  1.00000001e+00]
    Eigenvalues, after SVD: [-5.22427136e-09  9.99999993e-01  9.99999995e-01  9.99999995e-01
    9.99999996e-01  9.99999996e-01  9.99999996e-01  9.99999996e-01
    9.99999996e-01  9.99999997e-01  9.99999997e-01  9.99999997e-01
    9.99999997e-01  9.99999997e-01  9.99999998e-01  9.99999998e-01
    9.99999998e-01  9.99999998e-01  9.99999998e-01  9.99999998e-01
    9.99999998e-01  9.99999998e-01  9.99999998e-01  9.99999998e-01
    9.99999998e-01  9.99999998e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    9.99999999e-01  9.99999999e-01  9.99999999e-01  9.99999999e-01
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    1.00000000e+00  1.00000001e+00  1.00000001e+00  1.00000001e+00]
    """


if __name__ == "__main__":
    plac.call(main)