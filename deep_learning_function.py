import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

## miara niepewności
def entropy(labels, base=None):
    from math import log, e
    n_labels = len(labels)

    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <=1:
        return 0

    ent = 0.

    base = e if base is None else base
    for i in probs:
        ent -= i * log(i,base)
    return ent

#lables = [1,3,5,2,3,5,3,2,1,3,4,5]
#res_entropy = entropy(lables)


##===========================================


def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0000001, 1-0.0000001)
    return -y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)

#y_true = np.array([1,0,1,1,0,1,0])
#y_pred = np.array([0,0,1,1,0,1,1])

#res_binary_crossentropy = binary_crossentropy(y_true,y_pred)


##===========================================


def categorical_crossentropy(y_true, y_pred):
    return -np.sum(y_true*np.log(y_pred))

#y_true = np.array([1,0,0,0,0])
#y_pred = np.array([0.4, 0.3, 0.05, 0.05, 0.2])

#res_categorical_crossentropy = categorical_crossentropy(y_true, y_pred)


##===========================================

def gradient_descent(df=lambda w: 2 * w - 4, learning_rate=0.01, w_0 =-5, max_iters=10000, precision=0.000001):
    #licznik iteracji
    iters = 0

    #kontrola wartości kroku kolejnego spadku
    previous_step_size = 1

    weights = []
    while previous_step_size > precision and iters < max_iters:
        w_prev = w_0
        w_0 = w_0 - learning_rate * df(w_prev)
        previous_step_size = abs(w_0 - w_prev)
        iters += 1
        weights.append(w_0)
        print('Iter #{}: obecny punkt {}'.format(iters, w_0))
    print('Minimum globalne: {}'.format(w_0))
    return weights
        

#res_gradient_descent = gradient_descent()

#============================================

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    W2 = np.random.randn(n_h, n_y)
    return W1, W2

W1, W2 = initialize_parameters(2,2,1)

##===========================================


def forward_propagation(X, W1, W2):
    H = np.dot(X, W1)
    y_pred = np.dot(H, W2)
    return H, y_pred

def calculate_error(y_true, y_pred):
    return y_pred - y_true



np.random.seed(0)

X = np.array([[1.0, 0.7]])
y_true =np.array([1.80])

H, y_pred = forward_propagation(X, W1, W2)


#=============================================

def backpropagation(X, W1, W2, learning_rate=0.01, iters=1000, precision=0.0000001):

    H1, y_pred = forward_propagation(X, W1, W2)

    for i in range(iters):
        error = calculate_error(y_true, y_pred)
        W2 = W2 - learning_rate * error * H.T
        W1 = W1 - learning_rate * error * X.T * W2.T

        _, y_pred = forward_propagation(X, W1, W2)

        print('Iter {}, y_pred: {}, error: {}'.format(i, y_pred[0][0], calculate_error(y_true, y_pred)[0][0]))

        if abs(error) < precision:
            break
        
    return W1, W2



def predict(X, W1, W2):
    _, y_pred = forward_propagation(X, W1, W2)
    return y_pred


def build_model():

    # inicjalizacja wag
    W1, W2 = initialize_parameters(2, 2, 1)

    # propagacja wsteczna
    W1, W2 = backpropagation(X, W1, W2)

    model = {'W1' : W1, 'W2' : W2}

    return model

res = build_model()

########################


def get_raport(url, name, typ='biezace', date='0,0,0,1'):
    
    data = list()
    url = url+'/'+name+'/'+typ+','+date

    def clean_data(data):
        all_new_list = list()
        new_list = list()
        all_new_list.append(data[0].text.replace('\n',''))
        td = data[1][0].find_all('td')
        for i in range(len(td)):
            new_list.append(td[i].text.replace('\n',''))
        ahref = data[1][0].find_all('a')
        new_list.append(ahref[1]['href'].split(',')[0].split('/')[-1])
        all_new_list.append(new_list)
        return all_new_list

    def get_tr_from_soup(url):
        resp = requests.get(url)
        soup = bs(resp.text, 'html.parser')
        tabs = soup.find_all('table')
        tr = tabs[1].find_all('tr')
        return tr

    def check_pages(data, url):
        test = numpy.arange(50, 700, 50)
        for i in test:
            if len(data[1]) == i:
                last = int(url[-1])+1
                url = url[0:-1]
                url = url+str(last)
                tr = get_soup(url)
                for i in tr[2:]:
                    data[1].append(i)
        return data
          
    tr = get_tr_from_soup(url)
    try:
        today = tr[1]
        data.append(today)
        data.append(tr[2:])

        data = check_pages(data, url)
        data = clean_data(data)
        return data
    except:
        return 'brak banych'
