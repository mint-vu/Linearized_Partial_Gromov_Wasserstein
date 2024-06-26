#@nb.njit(fastmath=True)
def generate_d_square(d,n):
    x_list=[]
    for i in range(d):
        x = np.linspace(-1, 1, n)
        x_list.append(x)
    xx= np.meshgrid(*x_list,indexing='ij')
    data=np.zeros((n**d,d))
    for i in range(d):
        data[:,i]=xx[i].reshape(-1)
    return data


#@nb.njit(fastmath=True)
def generate_d_sphere(d,n,seed=20):
    np.random.seed(seed)
    N=n**d
    data=np.random.normal(size=(N,d))
    norm=np.sqrt(np.sum(np.square(data),1)).reshape(-1,1)
    data=data/norm
    return data