import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)

#mx = 241  
#my = 81 
#mx = 161  
mx = 161
my = 113
gamma = 1.4
k = 1/3 
ep = 1 # MUSCL
CFL = 0.50
r0 = 1
p0 = 1
#mach = 0.8
mach = 0.4
alpha = np.pi/180*(1.25) #degree
a0 = np.sqrt(gamma * p0 / r0)
u0 = a0 * mach * np.cos(alpha)
v0 = a0 * mach * np.sin(alpha)
#epsilon = 0.05 * a0
epsilon = -1


def read_mesh(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Skip the first two lines which contain the header information
    data_lines = lines[2:]
    
    x = np.zeros((mx, my))
    y = np.zeros((mx, my))
    
    idx = 0
    for j in range(my):
        for i in range(mx):
            x[i, j], y[i, j] = map(float, data_lines[idx].split())
            idx += 1
            
    return x, y
    
# Calculate geometric properties
def calc_geo(x,y):
    '''dxi_x = np.zeros((mx, my))
    dxi_y = np.zeros((mx, my))
    dxi   = np.zeros((mx, my))
    deta_x = np.zeros((mx, my))
    deta_y = np.zeros((mx, my))
    deta   = np.zeros((mx, my))'''
    vol = np.zeros((mx-1, my-1))
    edge = np.zeros((4, mx-1, my-1))
    vect = np.zeros((8, mx-1, my-1))

    # compute dn_bc and |dn_bc| along constant-xi boundaries
    for i in range(0, mx-1):
        for j in range(0, my-1):
            ip1 = i + 1 if i + 1 < mx else i + 1 - mx
            ip2 = i + 2 if i + 2 < mx else i + 2 - mx
            nv1 = np.array([y[i, j] - y[ip1, j], x[ip1, j] - x[i, j]])
            nv2 = np.array([y[i, j+1] - y[ip1, j+1], x[ip1, j+1] - x[i, j+1]])
            normv1 = np.linalg.norm(nv1)
            normv2 = np.linalg.norm(nv2)
            nv1 /= normv1
            nv2 /= normv2
            if normv1 == 0:
                raise ValueError(f"Negative normv1 r encountered at index ({i}, {j})")
            
            nh1 = np.array([y[i, j+1] - y[i, j], x[i, j] - x[i, j+1]])
            nh2 = np.array([y[ip1, j+1] - y[ip1, j], x[ip1, j] - x[ip1, j+1]])
            normh1 = np.linalg.norm(nh1)
            normh2 = np.linalg.norm(nh2)
            nh1 /= normh1
            nh2 /= normh2
            
            diag1 = np.array([x[ip1, j+1] - x[i, j], y[ip1, j+1] - y[i, j]])
            diag2 = np.array([x[i, j+1] - x[ip1, j], y[i, j+1] - y[ip1, j]])
            vol[i, j] = abs(np.linalg.det(np.array([diag1, diag2]))) * 0.5
            
            vect[:, i, j] = np.concatenate((nh1, nh2, nv1, nv2))
            edge[:, i, j] = [normh1, normh2, normv1, normv2]

    return vect, edge, vol
            
'''def calc_normal(x,y):
    for i in range(1, mx):
        for j in range(1, my):
            deta_x =  (y[i-1,j] - y[i,j])
            deta_y = -(x[i-1,j] - x[i,j])
            deta = np.sqrt(deta_x[i]**2 + deta_y[i]**2)

            dxi_x =  (y[i,j] - y[i,j-1])
            dxi_y = (x[i,j-1] - x[i,j])
            dxi = np.sqrt(dxi_x**2 + dxi_y**2)
    return dxi_x, dxi'''

def time_step(U, vect, edge, vol):
    

    dt = 1e-2 * np.ones((mx-1,my-1))
    dt0 = 1e-2 * np.ones((mx-1,my-1))
    for i in range(0, mx-1):
        for j in range(0, my-1):
            nh1 = vect[0:2, i, j]
            nh2 = vect[2:4, i, j]
            nv1 = vect[4:6, i, j]
            nv2 = vect[6:8, i, j]
            normh1 = edge[0, i, j]
            normh2 = edge[1, i, j]
            normv1 = edge[2, i, j]
            normv2 = edge[3, i, j]

            nh = 0.5 * (nh1 + nh2)
            nv = 0.5 * (nv1 + nv2)
            normh = 0.5*(normh1 + normh2)
            normv = 0.5*(normv1 + normv2)

            '''deta_x = nv1[0, i, j]
            deta_y = nv1[1, i,j]
            deta = normv1[i,j]

            dxi_x = nh1[0, i,j]
            dxi_y = nh1[1, i,j]
            dxi = normh1[i,j]'''
            
            r = U[0, i,j+1]
            u = U[1,i,j+1] / r
            v = U[2,i,j+1] / r
            E = U[3,i,j+1] 
            p = (gamma - 1.) * (E - 0.5 * r * (u ** 2 + v ** 2))
            a = np.sqrt(gamma * p / r)
            
            lambxi = (abs((nh[0]*u+nh[1]*v)) + a) * normh
            lambeta = (abs((nv[0]*u+nv[1]*v)) + a) * normv

            dt[i,j]  = CFL * vol[i, j] / (lambxi + lambeta)
    #print(dt)
    return dt


# a = 2
def time_advancement(u, vect, edge,vol):
    u1 = u.copy()
    u2 = u.copy()
    u3 = u.copy()
    u0 = u.copy()
    u4 = u.copy()

    '''u1[:, 2:-3, 2:-3] = u0[:, 2:-3, 2:-3] - dt / dx * 2 * method(u0)
                u2[:, 2:-3, 2:-3] = 3 / 4 * u0[:, 2:-3, 2:-3] + 1 / 4 * (u1[:, 2:-3, 2:-3] - dt / dx * 2 * method(u1))
                u3[:, 2:-3, 2:-3] = 1 / 3 * u0[:, 2:-3, 2:-3] + 2 / 3 * (u2[:, 2:-3, 2:-3] - dt / dx * 2 * method(u2))'''
    #Runge_Kutta
    '''p1 = total_flux(u0,vect, edge,vol)
    print('1')
    for j in range(0, my-1):
            for i in range(0, mx-1): 
                u1[:, i, j+1] = u0[:, i, j+1] - dt[i, j]  * p1[:, i, j+1]
    #u1 = interpolate(u1)
    #img(u1,1)
    p2 = total_flux(u1,vect, edge,vol)
    print('2')
    for j in range(0, my-1):
            for i in range(0, mx-1): 
                u2[:, i, j+1] = 3 / 4 * u0[:, i, j+1] + 1 / 4 * (u1[:, i, j+1] - dt[i, j] * p2[:, i, j+1])
    #u2 = interpolate(u2)
    #img(u2,2)
    p3 = total_flux(u2,vect, edge,vol)
    print('3')
    for j in range(0, my-1):
            for i in range(0, mx-1): 
                u3[:, i, j+1] = 1 / 3 * u0[:, i, j+1] + 2 / 3 * (u2[:, i, j+1] - dt[i, j]  * p3[:, i, j+1])
    #u3, _ = total_flux(u2,vect, edge,vol)'''

    dt = time_step(u, vect, edge, vol)
    p1 = total_flux(u0,vect, edge,vol)
    for j in range(0, my-1):
            for i in range(0, mx-1): 
                u1[:, i, j+1] = u0[:, i, j+1] - 1/4 * dt[i, j]  * p1[:, i, j+1]
    u1 = interpolate(u1)
    res = np.max(np.abs(u1 - u0))
    print(f"Iteration= {it}, {1}, max_dt= {np.max(dt)}, min_dt= {np.min(dt)} with res= {res}")

    dt = time_step(u1, vect, edge, vol)
    p2 = total_flux(u1,vect, edge,vol)
    for j in range(0, my-1):
            for i in range(0, mx-1): 
                u2[:, i, j+1] = u1[:, i, j+1] - 1/3 * dt[i, j] * p2[:, i, j+1]
    u2 = interpolate(u2)
    res = np.max(np.abs(u2 - u1))
    print(f"Iteration= {it}, {2}, max_dt= {np.max(dt)}, min_dt= {np.min(dt)} with res = {res}")

    dt = time_step(u2, vect, edge, vol)
    p3 = total_flux(u2,vect, edge,vol)
    for j in range(0, my-1):
            for i in range(0, mx-1): 
                u3[:, i, j+1] = u2[:, i, j+1] -1/2 * dt[i, j]  * p3[:, i, j+1]
    u3 = interpolate(u3)
    res = np.max(np.abs(u3 - u2))
    print(f"Iteration= {it}, {3}, max_dt= {np.max(dt)}, min_dt= {np.min(dt)} with res = {res}")

    dt = time_step(u3, vect, edge, vol)
    p4 = total_flux(u3,vect, edge,vol)
    for j in range(0, my-1):
            for i in range(0, mx-1): 
                u4[:, i, j+1] = u3[:, i, j+1] - 1.0 * dt[i, j]  * p4[:, i, j+1]
    u4 = interpolate(u4)
    res = np.max(np.abs(u4 - u3))
    print(f"Iteration= {it}, {4}, max_dt= {np.max(dt)}, min_dt= {np.min(dt)} with res = {res}")

    unew = u4
    return unew

def decoder(U):
    V = np.zeros(4)
    r = U[0]
    u = U[1] / r
    v = U[2] / r
    E = U[3] 
    p = (gamma - 1.) * (E - 0.5 * r * (u ** 2 + v ** 2))
    V = np.array([r, u, v, p])

    return V 

'''def encoder(V):
    r = V[0]
    u = V[1] 
    v = V[2] 
    p = V[3]

    U = np.zeros(4)
    U[0] = r
    U[1] = r*u
    U[2] = r*v
    U[3] = r * E'''


def flux_f(U):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    E = U[3] 
    p = (gamma - 1.) * (E - 0.5 * rho * (u ** 2 + v ** 2))

    F0 = np.array(rho * u)
    F1 = np.array(rho * u ** 2 + p)
    F2 = np.array(rho * u * v)
    F3 = np.array(u * (E + p))
    f = np.array([F0, F1, F2, F3])

    return f

def flux_g(U):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    E = U[3] 
    p = (gamma - 1.) * (E - 0.5 * rho * (u ** 2 + v ** 2))

    G0 = np.array(rho * v)
    G2 = np.array(rho * v ** 2 + p)
    G1 = np.array(rho * u * v)
    G3 = np.array(v * (E + p))
    g = np.array([G0, G1, G2, G3])
    
    return g

def entropy_cor(x):
    x = abs(x) if abs(x)>epsilon else ((x**2+epsilon**2)/(2*epsilon))
    return x


def char_Matrix(r, u, v, h,H, a, dU,n):
    A = 0
    dr = dU[0]
    du = dU[1] 
    dv = dU[2] 
    dp = dU[3]
    nx = n[0]
    ny = n[1]
    q = u*nx + v*ny
    dq = du*nx + dv*ny
    s = -u * ny + v * nx
    ds = -du * ny + dv * nx

    '''b2 = (gamma -1) / (a ** 2)
    b1= b2 * (u ** 2 + v ** 2) / 2''' 
    u1 = entropy_cor((q-a))
    u2 = entropy_cor(q)
    u3 = entropy_cor((q + a))

    l1 = np.array([1, u-a*nx, v-a*ny, H-a*q])
    l2 = np.array([0, -a*ny, a*nx, a*s])
    l3 = np.array([1, u, v, 0.5 * (u ** 2 + v ** 2)])
    l4 = np.array([1, u+a*nx, v+a*ny, H+a*q])

    A1 = 1/(2*a**2)*(dp - r*a*dq)
    A2 = r/a*ds
    A3 = 1/(a**2)*(a**2*dr - dp)
    A4 = 1/(2*a**2)*(dp + r*a*dq)

    lamb1 = u1
    lamb2 = u2
    lamb3 = u2
    lamb4 = u3

    A = (lamb1 * A1 * l1 + lamb2 * A2 * l2 + lamb3 * A3 * l3 + lamb4 * A4 * l4) 
    
    return A
    
'''def char_Matrix_y(r, u, v, h, a, dU,n):
    b2 = (gamma -1) / (a ** 2)
    b1= b2 * (u ** 2 + v ** 2) / 2 

    v1 = entropy_cor(v)
    v2 = entropy_cor(v - a)
    v3 = entropy_cor(v + a)

    Ly = np.array([
        [-b2 * u, 1 + b2 * u**2, b2 * u * v, -b2 * u],
        [1 - b1, b2 * u, b2 * v, -b2],
        [1/2 * (b1 + v / a), -1/2 * (b2 * u + 1 / a), -1/2 * b2 * v, 1/2 * b2],
        [1/2 * (b1 - v / a), -1/2 * (b2 * u - 1 / a), -1/2 * b2 * v, 1/2 * b2]
    ])
    Ry = np.array([[0, 1, 1, 1],
                    [1, 0, u, u],
                    [0, v, v - a,  v + a],
                    [u, 0.5 * (v ** 2 - u **2), h - a * v,  h + v * a]])
    lamby = np.array([[v1, 0, 0, 0],
                        [0, v1, 0, 0],
                        [0, 0, v2, 0],
                        [0, 0, 0, v3]])
    
    Ay = np.dot(Ry, lamby)
    Ay = np.dot(Ay, Ly)
    return Ay'''
'''l1 = [1 - b1, b2 * u, b2 * v, -b2],
l2 = [-b1 * v, b2 * u * v, 1 + b2 * v**2, -b2 * v],
l3 = [1/2 * (b1 + u / a), -1/2 * (b2 * u + 1 / a), -1/2 * b2 * v, 1/2 * b2],
l4 = [1/2 * (b1 - u / a), -1/2 * (b2 * u - 1 / a), -1/2 * b2 * v, 1/2 * b2]'''
'''Rx = np.array([[1, 0, 1, 1],
                [u, 0, u - a, u + a],
                [0, 1, v, v],
                [0.5 * (u ** 2 - v ** 2), v, h - a * u,  h + u * a]])'''

'''lambx = np.array([[u1, 0, 0, 0],
                    [0, u1, 0, 0],
                    [0, 0, u2, 0],
                    [0, 0, 0, u3]])

Ax = np.dot(Rx, lambx)
Ax = np.dot(Ax, Lx)'''

def MUSCL_limiter(r):
    return max(0, min(2*r, (r+1)/2, 2))

def Roe_Average(UL,UR):
        rL = UL[0]
        uL = UL[1] / rL
        vL = UL[2] / rL
        EL = UL[3] 
        pL = (gamma - 1.) * (EL - 0.5 * rL * (uL ** 2 + vL ** 2))
        aL = np.sqrt(gamma * pL / rL)
        HL = gamma / (gamma - 1.) * pL / rL + 0.5 * (uL ** 2 + vL **2)
        hL = gamma / (gamma - 1.) * pL / rL


        rR = UR[0]
        uR = UR[1] / rR
        vR = UR[2] / rR
        ER = UR[3] 
        pR = (gamma - 1.) * (ER - 0.5 * rR * (uR ** 2 + vR ** 2))
        aR = np.sqrt(gamma * pR / rR)
        HR = gamma / (gamma - 1.) * pR / rR + 0.5 * (uR ** 2 + vR **2)
        hR = gamma / (gamma - 1.) * pR / rR
        if pL<= 0:
            raise ValueError(f"Negative pL encountered at index ({i}, {j})")
        if pR<= 0:
            raise ValueError(f"Negative pR encountered at index ({i}, {j},{UR})")
        


        R = np.sqrt(rR / rL)
        rhat = R * rL
        uhat = (uL + R * uR) / (1. + R)
        vhat = (vL + R * vR) / (1. + R)
        Hhat = (HL + R * HR) / (1 + R)
        hhat = (hL + R * hR) / (1 + R)
        phat = (rhat * (gamma - 1) / gamma * (Hhat - 0.5 * (uhat ** 2 + vhat ** 2)))
        ahat = np.sqrt((gamma - 1) * (Hhat - 0.5 * (uhat ** 2 + vhat ** 2)))

        return rhat, uhat, vhat, phat, ahat, hhat, Hhat




def minmod(a, b):
    c = np.zeros(4)
    for i in range(4):
        if a[i] * b[i] > 0:
            c[i]=np.sign(a[i]) * min(abs(a[i]), abs(b[i]))
        else:
            c[i]=0
    return c

def solve_flux_xi(U,vect ,edge):
    '''r = U[0]
    u = U[1] / r
    v = U[2] / r
    E = U[3] 
    p = (gamma - 1.) * (E - 0.5 * r * (u ** 2 + v ** 2))
    a = np.sqrt(gamma * p / r)
    H = gamma / (gamma - 1.) * p / r + 0.5 * (u ** 2 + v **2)
    h = gamma / (gamma - 1.) * p / r
    check_negative_pressure(U)'''
    Hxi = np.zeros((4, mx, my-1))

    for j in range(0, my-1):
        for i in range(0, mx-1):
            ip1 = i + 1 if i + 1 < mx-1 else i + 2 - mx
            ip2 = i + 2 if i + 2 < mx-1 else i + 3 - mx
            #ip3 = i + 3 if i + 3 < mx else i + 3 - mx
            
            w = U[:, i, j+1] - U[:, i-1, j+1]
            wr = U[:, ip1, j+1] - U[:, i, j+1]
            wl = U[:, i-1, j+1] - U[:, i-2, j+1] 
            
            '''lambl = wl / w if w != 0 else 0
            lambr = wr / w if w != 0 else 0
            lambrr = 1 / lambr if lambr != 0 else 0
            lambll = 1/lambl if lambl != 0 else 0

            phil = MUSCL_limiter(lambl)
            phir = MUSCL_limiter(lambr)
            phirr = MUSCL_limiter(lambrr)
            phill = MUSCL_limiter(lambll)

            Ul = U[:, i, j] + ep / 4 * ((1-k)*phill*wl + (1+k)*phil*wr)
            Ur = U[:, i+1, j] - ep / 4 * ((1-k)*phirr*wl + (1+k)*phir*wr)'''
            
            UL = U[:, i-1, j+1]+0.5*minmod(wl,w)
            UR = U[:, i, j+1]-0.5*minmod(wr,w)
            #VL = decoder(UL)
            #(VL[3])
            rhat, uhat, vhat, phat, ahat, hhat, Hhat = Roe_Average(UL, UR)

            nh1 = vect[0:2,i, j]
            normh1 = edge[0, i, j]

            HL = (flux_f(UL) * nh1[0] + flux_g(UL) * nh1[1]) * normh1
            HR = (flux_f(UR) * nh1[0] + flux_g(UR) * nh1[1]) * normh1

            dU = decoder(UR) - decoder (UL)
            A = char_Matrix(rhat, uhat, vhat, hhat,Hhat, ahat, dU,nh1)
            Hxi[:, i,j] = 0.5*(HL+HR)-A* normh1 / 2
    return (Hxi)

def solve_flux_eta(U,vect ,edge):
    '''r = U[0]
    u = U[1] / r
    v = U[2] / r
    E = U[3] 
    p = (gamma - 1.) * (E - 0.5 * r * (u ** 2 + v ** 2))
    a = np.sqrt(gamma * p / r)
    H = gamma / (gamma - 1.) * p / r + 0.5 * (u ** 2 + v **2)
    h = gamma / (gamma - 1.) * p / r'''

    Heta = np.zeros((4, mx-1, my))

    for i in range(0, mx-1):
        for j in range(1, my-1):
            '''if j == my-2:
                w = U[:, i, j] - U[:, i, j-1]
                wr = U[:, i, j + 1] - U[:, i, j]
                wl = U[:, i, j-1] - U[:, i, j - 2] 
            else:'''
            w = U[:, i, j + 1] - U[:, i, j]
            wr = U[:, i, j + 2] - U[:, i, j + 1]
            wl = U[:, i, j] - U[:, i, j - 1] 
        
            '''lambl = wl / w
            lambr = wr / w
            lambrr = 1 / lambr
            lambll = 1/lambl

            phil = MUSCL_limiter(lambl)
            phir = MUSCL_limiter(lambr)
            phirr = MUSCL_limiter(lambrr)
            phill = MUSCL_limiter(lambll)

            Ul = U[:, i, j] + ep / 4 * ((1-k)*phill*wl + (1+k)*phil*wr)
            Ur = U[:, i, j+1] - ep / 4 * ((1-k)*phirr*wl + (1+k)*phir*wr)'''
            
            UL = U[:, i, j]+0.5*minmod(wl,w)
            UR = U[:, i, j+1]-0.5*minmod(wr,w)
            
            rhat, uhat, vhat, phat, ahat, hhat, Hhat = Roe_Average(UL, UR)

            nv1 = vect[4:6,i, j]
            normv1 = edge[2, i, j]

            HL = (flux_f(UL) * nv1[0] + flux_g(UL) * nv1[1]) * normv1
            HR = (flux_f(UR) * nv1[0] + flux_g(UR) * nv1[1]) * normv1

            dU = decoder(UR) - decoder (UL)
            A = char_Matrix(rhat, uhat, vhat, hhat,Hhat, ahat, dU,nv1)
            Heta[:, i,j] = 0.5*(HL+HR)-A* normv1 / 2
    return (Heta)

def total_flux(U,vect,edge,vol):
    P =np.zeros((4,mx-1,my+1))
    Hxi = solve_flux_xi(U,vect,edge)
    Heta = solve_flux_eta(U,vect,edge)
    Hxi[:, mx-1, :] = Hxi[:, 0, :]
    Uwall, Heta = bc_wall(U,Heta)
    Ufar, Heta = bc_inflow(U, Heta)
    #U = interpolate(U,Uwall,Ufar)

    for i in range(0, mx-1):
        for j in range(0, my-1):
            P[:, i, j+1] = (Hxi[:, i+1, j] - Hxi[:, i, j] + Heta[:, i, j+1] - Heta[:, i, j])/vol[i,j]
    #print(P[0,:,0])
    return P

def interpolate(U):
    H = np.zeros_like(U)
    Uwall, _ = bc_wall(U, H)
    Ufar, _ = bc_inflow(U, H)
    #U[:, 1:-3, 1] = 2 * U[:, 1:-3, 2] - U[:, 1:-3, 3]
    #U[:, 1:-1, 0] = 2*U[:, 1:-1, 1] - U[:, 1:-1, 2]
    #U[:, :, -3] = 2*U[:, :, -4] - U[:, :, -5]

    U[:, :, 0] = 2*Uwall- U[:,:,1]
    #U[:, :, 0] = Uwall
    U[:, :, -1] = 2*Ufar - U[:, :, -2]
    #U[:, :, -1] = Ufar

    #U[:, 1, 1:-1] = 2 * U[:, 2, 1:-1] - U[:, 3, 1:-1]
    #U[:, -3, 1:-1]= 2 * U[:, -4, 1:-1] - U[:, -5, 1:-1]
    #U[:, -2, 1:-1]= 2 * U[:, -3, 1:-1] - U[:, -4, 1:-1]
    return U

def bc_wall(U, Heta):
    j = 0
    Uwall = np.zeros((4,mx-1))
    for i in range(0, mx-1):
        ip1 = i + 1 if i + 1 < mx else i + 1 - mx
        ip2 = i + 2 if i + 2 < mx else i + 2 - mx
        #U_ghost = 1.5 * U[:, i, j+1] - 0.5*U[:, i, j+2]
        U_ghost = U[:, i, j+1]
        r_ghost = U_ghost[0]
        u_ghost = U_ghost[1] / r_ghost
        v_ghost = U_ghost[2] / r_ghost
        E_ghost = U_ghost[3]    
        p_ghost = (gamma - 1.) * (E_ghost - 0.5 * r_ghost * (u_ghost ** 2 + v_ghost ** 2))
        if p_ghost<0:
            raise ValueError(f"Negative ghost pressure encountered at index ({i}, {j}): p={p_ghost}")

        a_ghost = np.sqrt(gamma * p_ghost / r_ghost)

        deta_x =  (y[i,j] - y[ip1,j])
        deta_y = -(x[i,j] - x[ip1,j])
        deta = np.sqrt(deta_x**2 + deta_y**2)
        if deta == 0:
            raise ValueError(f"Negative eta r encountered at index ({i}, {j})")

        nx = deta_x / deta 
        ny = deta_y / deta 

        Vn_ghost = u_ghost*nx+v_ghost*ny

        u = u_ghost*ny**2-v_ghost*nx*ny
        v = -u_ghost*nx*ny + v_ghost*nx**2
        r = np.power(((gamma-1)**2*(Vn_ghost-2*a_ghost/(gamma-1))**2*np.power(r_ghost,gamma)/(4*gamma*p_ghost)),1/(gamma-1))
        p = p_ghost * np.power(r,gamma) / np.power(r_ghost,gamma)

        Uwall[0,i]  = r
        Uwall[1,i]  = u*r
        Uwall[2,i]  = v*r
        Uwall[3,i]  = p/(gamma-1)+0.5*r*(u**2+v**2)
        if r <= 0 :
            raise ValueError(f"Negative bc wall r encountered at index ({i}, {j}): r={r}")

        #print(deta_x,deta_y)
        Heta[:,i,j] = deta * (flux_f(Uwall[:,i])*nx+flux_g(Uwall[:,i])*ny)
    return Uwall, Heta

def bc_inflow(U, Heta):
    j = my-1
    Ufar = np.zeros((4,mx-1))

    for i in range(0, mx-1):
        ip1 = i + 1 if i + 1 < mx else i + 1 - mx
        ip2 = i + 2 if i + 2 < mx else i + 2 - mx
        U_ghost = 1.5*U[:, i, j] - 0.5*U[:, i, j-1]
        #U_ghost = U[:, i, j]
        #print(U_ghost)
        r_ghost = U_ghost[0]
        u_ghost = U_ghost[1] / r_ghost
        v_ghost = U_ghost[2] / r_ghost
        E_ghost = U_ghost[3]    
        p_ghost = (gamma - 1.) * (E_ghost - 0.5 * r_ghost * (u_ghost ** 2 + v_ghost ** 2))
        a_ghost = np.sqrt(gamma * p_ghost / r_ghost)

        deta_x =  (y[i,j] - y[ip1, j])
        deta_y = -(x[i,j] - x[ip1, j])
        deta = np.sqrt(deta_x**2 + deta_y**2)
        if deta == 0:
            raise ValueError(f"Negative eta r encountered at index ({i}, {j})")


        #print(deta)
        nx = deta_x / deta
        ny = deta_y / deta

        Vn0 = u0*nx+v0*ny
        Vt0 = u0*ny-v0*nx

        Vn_ghost = u_ghost*nx+v_ghost*ny
        Vt_ghost = u_ghost*ny-v_ghost*nx

        Vn = 0
        a = 0
        r = 0
        p = 0
        Vt = 0

        if Vn0 <= 0:
            Vn = 0.5 * (Vn0 + Vn_ghost) + 1/(gamma -1 ) * (a_ghost - a0)
            a = (gamma-1)/4 * (Vn_ghost - Vn0) + 0.5 * (a_ghost + a0)
            r = np.power((a**2 * np.power(r0,gamma) / (gamma * p0)),(1/(gamma-1)))
            #p = p0 * np.power(r,gamma) / np.power(r0,gamma)
            p = 1 / gamma * a**2 *r
            Vt = Vt0

        if Vn0 > 0:
            Vn = 0.5 * (Vn0 + Vn_ghost) + 1/(gamma -1 ) * (a_ghost - a0)
            a = (gamma-1)/4 * (Vn_ghost - Vn0) + 0.5 * (a_ghost + a0)
            r = np.power((a**2 * np.power(r_ghost,gamma) / (gamma * p_ghost)),(1/(gamma-1)))
            #p = p_ghost * np.power(r,gamma) / np.power(r_ghost,gamma)
            p = 1 / gamma * a**2 *r
            Vt = Vt_ghost

        u = nx*Vn + ny*Vt
        v = ny*Vn - nx*Vt

        Ufar[0,i]  = r
        Ufar[1,i]  = u*r
        Ufar[2,i]  = v*r
        Ufar[3,i]  = p/(gamma-1)+0.5*r*(u**2+v**2)

        if r<= 0:
                raise ValueError(f"Negative bc wall r encountered at index ({i}, {j}): r={r}")


        
        Heta[:,i,j] = deta * (flux_f(Ufar[:,i])*nx+flux_g(Ufar[:,i])*ny)
        
        
    return Ufar, Heta

'''def bc_outflow(U):
    i = 0
    for j in range(0, my-1):
        U_ghost = 2 * U[:, i+1, j] - U[:, i+2, j]
        r_ghost = U_ghost[0]
        u_ghost = U_ghost[1] / r_ghost
        v_ghost = U_ghost[2] / r_ghost
        E_ghost = U_ghost[3]    
        p_ghost = (gamma - 1.) * (E_ghost - 0.5 * r_ghost * (u_ghost ** 2 + v_ghost ** 2))
        a_ghost = np.sqrt(gamma * p_ghost / r_ghost)

        dxi_x =  (y[i,j+1] - y[i,j])
        dxi_y = (x[i,j] - x[i,j+1])
        dxi = np.sqrt(dxi_x**2 + dxi_y**2)

        nx = dxi_x / dxi
        ny = dxi_y / dxi

        Vn0 = u0*nx+v0*ny
        Vt0 = u0*ny-v0*nx

        Vn_ghost = u_ghost*nx+v_ghost*ny
        Vt_ghost = u_ghost*ny-v_ghost*nx

        Vn = 0.5 * (Vn0 + Vn_ghost) + 1/(gamma -1 ) * (a_ghost - a0)
        a = (gamma-1)/4 * (Vn_ghost - Vn0) + 0.5 * (a_ghost + a0)
        r = np.power((a**2 * np.power(r_ghost,gamma) / (gamma * p_ghost)),(1/(gamma-1)))
        p = p_ghost * np.power(r,gamma) / np.power(r_ghost,gamma)
        Vt = Vt_ghost

        u = nx*Vn + ny*Vt
        v = ny*Vn - nx*Vt

        U[0,i,j]  = r
        U[1,i,j]  = u*r
        U[2,i,j]  = v*r
        U[3,i,j]  = p/(gamma-1)+0.5*r*(u**2+v**2)

    i = mx-1
    for j in range(0, my-1):
        U_ghost = 2 * U[:, i-1, j] - U[:, i-2, j]
        r_ghost = U_ghost[0]
        u_ghost = U_ghost[1] / r_ghost
        v_ghost = U_ghost[2] / r_ghost
        E_ghost = U_ghost[3]    
        p_ghost = (gamma - 1.) * (E_ghost - 0.5 * r_ghost * (u_ghost ** 2 + v_ghost ** 2))
        a_ghost = np.sqrt(gamma * p_ghost / r_ghost)

        dxi_x =  (y[i,j+1] - y[i,j])
        dxi_y = (x[i,j] - x[i,j+1])
        dxi = np.sqrt(dxi_x**2 + dxi_y**2)

        nx = dxi_x / dxi
        ny = dxi_y / dxi

        Vn0 = u0*nx+v0*ny
        Vt0 = u0*ny-v0*nx

        Vn_ghost = u_ghost*nx+v_ghost*ny
        Vt_ghost = u_ghost*ny-v_ghost*nx

        Vn = 0.5 * (Vn0 + Vn_ghost) + 1/(gamma -1 ) * (a_ghost - a0)
        a = (gamma-1)/4 * (Vn_ghost - Vn0) + 0.5 * (a_ghost + a0)
        r = np.power((a**2 * np.power(r_ghost,gamma) / (gamma * p_ghost)),(1/(gamma-1)))
        p = p_ghost * np.power(r,gamma) / np.power(r_ghost,gamma)
        Vt = Vt_ghost

        u = nx*Vn + ny*Vt
        v = ny*Vn - nx*Vt

        U[0,i,j]  = r
        U[1,i,j]  = u*r
        U[2,i,j]  = v*r
        U[3,i,j]  = p/(gamma-1)+0.5*r*(u**2+v**2)
    return U'''



def ic(U):
    H = np.zeros_like(U)
    for i in range(mx-1):
        for j in range(my+1):

            U[0,i,j]  = r0
            U[1,i,j]  = u0*r0
            U[2,i,j]  = v0*r0
            U[3,i,j]  = p0/(gamma-1)+0.5*r0*(u0**2+v0**2)
    #U =interpolate(U)
    #print(U[1,:,0])
    #U, _ = bc_wall(U, H)
    #print(U[1,:,0])
    #U, _ = bc_inflow(U, H)
    #print(U[1,:,0])
    check_negative_pressure(U)
    return U


def write_tecplot(filename, x, y, U, step):
    with open(filename, 'w') as f:
        f.write('VARIABLES = "x", "y", "rho", "u", "v", "p"\n')
        f.write(f'ZONE T="Step {step}", I={mx}, J={my}, F=POINT\n')
        for j in range(my):
            for i in range(mx):
                rho = U[0, i, j]
                u = U[1, i, j] / rho
                v = U[2, i, j] / rho
                E = U[3, i, j]
                p = (gamma - 1.) * (E - 0.5 * rho * (u ** 2 + v ** 2))
                f.write(f"{x[i, j]} {y[i, j]} {rho} {u} {v} {p}\n")

def check_negative_pressure(U):
    for i in range(mx-1):
        for j in range(my+1):
            r = U[0, i, j]
            u = U[1, i, j] / r
            v = U[2, i, j] / r
            E = U[3, i, j]
            p = (gamma - 1.) * (E - 0.5 * r * (u ** 2 + v ** 2))
            if r<=0:
                raise ValueError(f"Negative rho encountered at index ({i}, {j}): r={r}")
            if p <= 0:
                raise ValueError(f"Negative pressure encountered at index ({i}, {j}): p={p}")

def center2node(U):
    Unode = np.zeros((4,mx, my))

    # Interior nodes
    for j in range(1, my-1):  # Interior nodes (indices from 1 to jdim-1 in 0-based indexing)
        for i in range(mx):  # Node's index is used
            if i == 0:
                k = mx - 2
                t = 0
            elif i == mx-1:
                k = mx - 2
                t = 0
            else:
                k = i - 1
                t = i
            Unode[:, i, j] = 0.25 * (U[:, t, j + 1] + U[:, t, j] + U[:, k, j + 1] + U[:, k, j])

    # Boundary nodes
    for i in range(mx):
        if i == 0:
            k = mx - 2
            t = 0
        elif i == mx-1:
            k = mx - 2
            t = 0
        else:
            k = i - 1
            t = i
        Unode[:, i, 0] = 0.5 * (U[:, t, 0] + U[:, k, 0])
        Unode[:, i, my-1] = 0.5 * (U[:, t, my] + U[:, k, my])

    return Unode

def img(U,k):
    plt.figure(figsize=(10, 5))
    plt.contourf(x, y, U[0, :, :])
    plt.colorbar()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.4, 0.4)
    plt.savefig(f"img{k}.png",dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.contourf(x, y, U[0, :, :])
    plt.colorbar()
    #plt.xlim(-0.1, 1.1)
    #plt.ylim(-0.4, 0.4)
    plt.savefig(f"img_full{k}.png",dpi=300)
    plt.close()
    #print('img')

if __name__ == '__main__':
    U = np.zeros((4,mx-1,my+1))
    #dt = 1e-3 * np.ones((mx,my))
    global x,y
    #x, y = read_mesh('grid_6000.dat')
    x, y = read_mesh('grid.dat')
    #x, y = x[:-1,:],y[:-1,:]
    print("Mesh readed")
    #print(x[:,0],y[:,0])

    vect, edge, vol = calc_geo(x, y)
    plt.figure(figsize=(10, 5))
    for j in range(my):
        plt.plot(x[:, j], y[:, j], 'r-', linewidth=0.03)  
    for i in range(mx):
        plt.plot(x[i, :], y[i, :], 'r-', linewidth=0.03)  

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grid Visualization')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.grid(True)
    plt.savefig('grid111',dpi=3000)
    plt.close()
    
    U = ic(U)
    print("Initialized")
    unode = center2node(U)
    img(unode,0)
    
    plt.close()
    tEnd = 1.0
    t = 0.
    it = 0.
    res =1.

    while it < 100000 and res > 1e-6:
        H = np.zeros_like(U)
        U0 = U.copy()
        '''if it < 10:
            dt = 1e-2*np.ones((mx,my))
        if it >= 10:'''
        #dt = 1e-2*np.ones((mx,my))
        #dt = time_step(U, vect, edge)
        #print(U[0,:,0])
        U = time_advancement(U0, vect, edge, vol)
        U = interpolate(U)
        #U = bc_wall(U)
        #U,_ = bc_inflow(U, H)
        #U = bc_outflow(U)
        #check_negative_pressure(U)
        res = np.max(np.abs(U - U0))
        #print(U[0,0,:])

        #t = t + dt
        it = it + 1
        print(f"Iteration= {it}, with residual= {res}")

        if it % 200 == 0:
            Unode = center2node(U)
            plt.figure(figsize=(10, 5))
            plt.contourf(x, y, Unode[0, :, :])
            plt.colorbar()
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.4, 0.4)
            plt.savefig(f"img{it}.png",dpi=300)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.contourf(x, y, Unode[0, :, :])
            plt.colorbar()
            #plt.xlim(-0.1, 1.1)
            #plt.ylim(-0.4, 0.4)
            plt.savefig(f"img_full{it}.png",dpi=300)
            plt.close()
        if it % 200 == 0:
            Unode = center2node(U)
            write_tecplot(f"output_{it}.dat", x, y, Unode, it)
    print("Finished")


