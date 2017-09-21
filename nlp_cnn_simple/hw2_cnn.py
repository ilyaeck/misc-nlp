#################################################
# ================ DO NOT MODIFY ================
#################################################
import sys
import math
import numpy as np
import pickle

# convolution window size
width = 2

# number of filters
F = 100

# learning rate
alpha = 1e-1

# vocabsize: size of the total vocabulary
# the text in the input file will be transformed into respective
# positional indices in the vocab dictionary
# as the input for the forward and backward algorithm
# e.g. if vocab = {'hello': 0, 'world': 1} and the training data is
# "hello hello world hello world",
# the input to the forward and backward algorithm will be [0, 0, 1, 0, 1]
vocabsize = 10000
vocab = {}

np.random.seed(1)

# U and V are weight vectors of the hidden layer
# U: a matrix of weights of all inputs for the first
# hidden layer for all F filters
# where each filter has the size of vocabsize by width
# U[i, j, k] represents the weight of filter u_j
# for word with vocab[word] = i when the word is
# in the position offset k of the sliding window
# e.g. in our earlier example of "hello hello world hello world",
# if the window size is 4 and we are looking at the first sliding window
# of the 9th filter, the weight for the last "hello" will be U[0, 8, 3]
U = np.random.normal(loc=0, scale=0.01, size=(vocabsize, F, width))

# V: the weight vector of the F filter outputs (after max pooling)
# that will produce the output, i.e. o = sigmoid(V*h)
V = np.random.normal(loc=0, scale=0.01, size=(F))


def sigmoid(x):
    """
    helper function that computes the sigmoid function
    """
    return 1. / (1 + math.exp(-x))


def read_vocab(filename):
    """
    helper function that builds up the vocab dictionary for input transformation
    """
    file = open(filename)
    for line in file:
        cols = line.rstrip().split("\t")
        word = cols[0]
        idd = int(cols[1])
        vocab[word] = idd
    file.close()


def read_data(filename):
    """
    :param filename: the name of the file
    :return: list of tuple ([word index list], label)
    as input for the forward and backward function
    """
    data = []
    file = open(filename)
    for line in file:
        cols = line.rstrip().split("\t")
        label = int(cols[0])
        words = cols[1].split(" ")
        w_int = []
        for w in words:
            # skip the unknown words
            if w in vocab:
                w_int.append(vocab[w])
        data.append((w_int, label))
    file.close()
    return data


def train():
    """
    main caller function that reads in the names of the files
    and train the CNN to classify movie reviews
    """
    vocabFile = sys.argv[2]
    trainingFile = sys.argv[3]
    testFile = sys.argv[4]

    read_vocab(vocabFile)
    training_data = read_data(trainingFile)
    test_data = read_data(testFile)

    for i in range(50):
        # confusion matrix showing the accuracy of the algorithm
        confusion_training = np.zeros((2, 2))
        confusion_validation = np.zeros((2, 2))
        sys.stdout.write('Training epoch with full data batch')
        for (data, label) in training_data:
            # back propagation to update weights for both U and V
            backward(data, label)

            # calculate forward and evaluate
            prob = forward(data)["prob"]
            pred = 1 if prob > .5 else 0
            confusion_training[pred, label] += 1
            sys.stdout.write('.')  # Vizualize progress
            sys.stdout.flush()

        for (data, label) in test_data:
            # calculate forward and evaluate
            prob = forward(data)["prob"]
            pred = 1 if prob > .5 else 0
            confusion_validation[pred, label] += 1

        print("Epoch: {}\tTrain accuracy: {:.3f}\tDev accuracy: {:.3f}"
            .format(
            i,
            np.sum(np.diag(confusion_training)) / np.sum(confusion_training),
            np.sum(np.diag(confusion_validation)) / np.sum(confusion_validation)))


#################################################
# ========= IMPLEMENT FUNCTIONS BELOW ===========
#################################################

def forward(word_indices):
    """
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :return: a result dictionary containing 3 fields -

    result['prob']:
    output of the CNN algorithm. predicted probability of 1

    result['h']:
    the hidden layer output after max pooling, h = [h1, ..., hF]

    result['hid']:
    argmax of F filters, e.g. j of x_j
    e.g. for the ith filter u_i, tanh(word[hid[i], hid[i] + width]*u_i) = max(h_i)
    """


    h = np.zeros(F, dtype=float)
    hid = np.zeros(F, dtype=int)
    prob = 0

    # step 1. compute h and hid
    # loop through the input data of word indices and
    # keep track of the max filtered value h_i and its position index x_j
    # h_i = max(tanh(weighted sum of all words in a given window)) over all windows for u_i
    """
    Type your code below
    """
    # First (hidden) layer
    n_in = len(word_indices)
    n_out = n_in - width + 1  #number of outputs after 1 convolution pass
    #First, precompute actual inputs of U filters
    input_chunks = np.zeros((n_out, width), dtype=int)
    for i in range(n_out):
        # Select the inputs coresponding to the current window position X[i..(i+width-1)]
        # and precompute chunks of conatenated inputs for each filter window position.
        for k in range(width):
            # Optimization: store only vocab indices of words, as opposed to 1-hot vectors.
            input_chunks[i,k] = word_indices[i+k]

    #Now, convolve with all filters, one by one.
    for j in range(F):   # for each filter
        h[j] = -1   # will hold max-pooling result for filter j
        hid[j] = 0
        filter_j = np.reshape(U[:,j,:], [width*vocabsize, 1])

        # print("Convolving input with filter {}".format(j))

        for i in range(n_out):  # Compute each y_i in the output of the first layer
            y_i = np.tanh(x_dot_u(input_chunks[i,:], j))
            # Update max over outputs y_l, for filter j.
            if y_i > h[j]:
                h[j] = max(h[j], y_i)
                hid[j] = i

    # step 2. compute probability
    # once h and hid are computed, compute the probabiliy by sigmoid(h^T*V)
    """
    Type your code below
    """
    prob = sigmoid(np.dot(V, h))

    # step 3. return result
    return {"prob": prob, "h": h, "hid": hid}


def backward(word_indices, true_label):
    """
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :param true_label: true label (0, 1) of the movie reviews
    :return: None

    update weight matrix/vector U and V based on the loss function
    """
    global U, V
    pred = forward(word_indices)
    prob = pred["prob"]
    h = pred["h"]
    hid = pred["hid"]

    # update U and V here
    # loss_function = y * log(o) + (1 - y) * log(1 - o)
    #               = true_label * log(prob) + (1 - true_label) * log(1 - prob)
    # to update V: V_new = V_current + d(loss_function)/d(V)*alpha
    # to update U: U_new = U_current + d(loss_function)/d(U)*alpha
    # Make sure you only update the appropriate argmax term for U
    """
    Type your code below
    """
    delta_U = calc_analytical_gradients_U(U, word_indices, h, hid, true_label)*alpha
    delta_V = calc_analytical_gradients_V(V, word_indices, h, true_label)*alpha
    # Add the delta becuase we are MAXIMIZING the objective function
    V = V + delta_V
    # For U, given the argmax index i, update only the slices of U corresponding to
    # [word_indices[i],..,word_indices[i+k-1]]
    for j_filter in range(F):
        i = hid[j_filter]
        for k in range(width):
            slice_idx = word_indices[i+k]
            U[slice_idx, j_filter, k] = delta_U[j_filter, k]
    return


def calc_numerical_gradients_V(V_obsolete, word_indices, true_label):
    """
    :param true_label: true label of the data
    :param V: weight vector of V
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :return V_grad:
    V_grad =    a vector of size length(V) where V_grad[i] is the numerical
                gradient approximation of V[i]
    """
    # you might find the following variables useful
    x = word_indices
    y = true_label
    eps = 1e-4
    V_grad = np.zeros(F, dtype=float)

    """
    Type your code below
    """
    # Compute numerical partial derivatives for each v_i in V, one at a time
    sys.stdout.write("Computing numerical V-gradient")
    for v_i, vgrad_i in np.nditer([V, V_grad], op_flags=['readwrite','readwrite']):
        v_i[...] = v_i + eps  # Alter v_i by adding a positive delta
        # Make a prediction and compute loss
        predicted_prob = forward(word_indices)["prob"]

        # loss_function = y * log(o) + (1 - y) * log(1 - o)
        loss_1 =  true_label*np.log(predicted_prob) + (1-true_label)*np.log(1-predicted_prob)
        v_i[...] = v_i - 2*eps  # Alter v_i by adding a negative delta

        # Make another prediction and compute loss
        predicted_prob = forward(word_indices)["prob"]
        loss_2 =  true_label*np.log(predicted_prob) + (1-true_label)*np.log(1-predicted_prob)

        vgrad_i[...] = (loss_1 - loss_2) / (eps+eps)  # apply finite differencing
        v_i[...] = v_i + eps  # undo modifications

        sys.stdout.write('.') # Just to visualize progress
        sys.stdout.flush()
    print("done.")
    return V_grad


def calc_numerical_gradients_U(U_obsolete, word_indices, true_label):
    """
    :param U: weight matrix of U
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :param true_label: true label of the data
    :return U_grad:
    U_grad =    a matrix of dimension F*width where U_grad[i, j] is the numerical
                approximation of the gradient for the argmax of
                each filter i at offset position j
    """
    # you might find the following variables useful
    x = word_indices
    y = true_label
    eps = 1e-4

    pred = forward(x)
    prob = pred["prob"]
    h = pred["h"]
    hid = pred["hid"]
    U_grad = np.zeros((F, width))

    """
    Type your code below
    """
    sys.stdout.write("Computing numerical U-gradient")
    # Compute numerical partial derivatives for each v_i in V, one at a time
    for j_filter in range(F):
        # Given the argmax index i, update only the slices of U corresponding to
        # [word_indices[i],..,word_indices[i+k-1]]
        i = hid[j_filter]
        for k in range(width):  # Alter u_ij by adding a positive delta
            slice_idx = word_indices[i+k]
            U[slice_idx, j_filter, k] += eps
            # Make a prediction and compute loss
            predicted_prob = forward(word_indices)["prob"]
            # loss_function = y * log(o) + (1 - y) * log(1 - o)
            loss_1 =  true_label*np.log(predicted_prob) + (1-true_label)*np.log(1-predicted_prob)
            U[slice_idx, j_filter, k] -= 2* eps  # Alter u_ij by adding a negative delta

            # Make another prediction and compute loss
            predicted_prob = forward(word_indices)["prob"]
            loss_2 =  true_label*np.log(predicted_prob) + (1-true_label)*np.log(1-predicted_prob)

            # Finally, compute grad and undo modifications to U
            U_grad[j_filter, k] = (loss_1 - loss_2) / (eps+eps)  # apply finite differencing
            U[slice_idx, j_filter, k] += eps

            sys.stdout.write('.') # Just to visualize progress
            sys.stdout.flush()
    print("done.")
    return U_grad

# Compute dot product between a chunk of conctenated input words vectors x_i
# and filter u_j
def x_dot_u(input_chunk_word_idxs, u_idx):
    """
    :param input_chunk_word_idxs: vocab indices of words in this input chunk
    :param u_idx: filter index
    """
    # Dot product with one-hot vectors is tantamount to selecting a slice of U
    # correponding to a given word vocab index, so we optiize the operation.
    dotprod = 0
    for k,word_idx in enumerate(input_chunk_word_idxs):
        dotprod += U[word_idx,u_idx,k]
    return dotprod


def calc_analytical_gradients_V(V_obsolete, word_indices, h, true_label):
    """
    grad_V = (y-sigmoid(V*h))*h
    """
    grad_V = (true_label - sigmoid(np.dot(V, h)))*h
    return grad_V

def calc_analytical_gradients_U(U_obsolete, word_indices, h, hid, true_label):
    """
    dL/du_j = [true_label - sigmoid(V*h)] * V * [1 - tanh^2(dot(x_m,u_j))] * {x_m}^T
        where x_m is the maxpooling "winner" for u_j.
    In practice, we do not multiply
    """
    common_coeff = true_label - sigmoid(np.dot(V, h))
    U_grad = np.zeros((F, width))
    argmax_input_chunks = np.zeros(width, dtype=int)
    for j_filter in range(F):
        i_argmax = hid[j_filter]
        for k in range(width): # Collect "argmax" word indices for this filter
            argmax_input_chunks[k] = word_indices[i_argmax+k]
        tanh_val = np.tanh(x_dot_u(argmax_input_chunks, j_filter)) ## This is te term that depends on U.
        coeff = common_coeff*V[j_filter]*(1-tanh_val*tanh_val)
        U_grad[j_filter,:] = coeff # All kernel coefficients are same??
    return U_grad


def check_gradient():
    """
    :return (diff in V, diff in U)
    Calculate numerical gradient approximations for U, V and
    compare them with the analytical values
    check gradient accuracy; for more details, cf.
    http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
    """
    x = []
    for i in range(100):
        x.append(np.random.randint(vocabsize))
    y = 1

    pred = forward(x)
    prob = pred["prob"]
    h = pred["h"]
    hid = pred["hid"]

    """
    Update 0s below with your calculations
    """
    # check V
    # compute analytical gradient
    ana_grad_V = calc_analytical_gradients_V(V, x, h, y)
    # compute numerical gradient
    numerical_grad_V = calc_numerical_gradients_V(V, x, y)
    # compare the differences
    sum_V_diff = sum((numerical_grad_V - ana_grad_V) ** 2)

    # check U
    # compute analytical and numerical gradients and compare their differences
    ana_grad_U = calc_analytical_gradients_U(U, x, h, hid, y)
    numerical_grad_U = calc_numerical_gradients_U(U, x, y)
    sum_U_diff = sum(sum((numerical_grad_U - ana_grad_U) ** 2))

    print("V diff: {:.8f}, U diff: {:.8f} (these should be close to 0)"
          .format(sum_V_diff, sum_U_diff))



#################################################
# ================ DO NOT MODIFY ================
#################################################

def load_gradient_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.U, data.word_indices, data.true_label


def load_gradient_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.U, data.word_indices, data.true_label


def load_forward_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.word_indices

def load_backward_vars():
    with open('grad.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)

    return data.word_indices, data.true_label

# run the entire file with:
# python hw2_cnn.py -t vocab.txt movie_reviews.train movie_reviews.dev
#
if __name__ == "__main__":
    if sys.argv[1] == "-t":
        check_gradient()
        train()
    elif sys.argv[1] == "-g":
        U, word_indices, true_label = load_gradient_vars()
        calc_numerical_gradients_U(U, word_indices, true_label)
    elif sys.argv[1] == "-f":
        word_indices = load_forward_vars()
        forward(word_indices)
    elif sys.argv[1] == "-b":
        word_indices, true_label = load_backward_vars()
        backward(word_indices, true_label)
    else:
        print("Usage: python hw2_cnn.py -t vocab.txt movie_reviews.train movie_reviews.dev")
