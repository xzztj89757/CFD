import matplotlib.pyplot as plt
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import math
np.set_printoptions(threshold=np.inf)

# Constants
XMAX = 300
YMAX = 100

# Function to initialize grid
'''def initialize_grid(IMAX, JMAX, thickness=0.12):
    # Memory allocation and initialization
    for i in range(IMAX + 2):
        for j in range(JMAX):
            x[i][j] = 0.001
            y[i][j] = 0.001
            P[i][j] = 0.0
            Q[i][j] = 0.0
            
    return x, y, P, Q'''

# Constants
IMAX = 241  # Number of nodes in ksi direction (IMAX-1 divisions)
JMAX = 81   # Number of nodes in eta direction (JMAX-1 divisions)
thickness = 0.12
PI = math.pi

# Initialize grid arrays
x = np.zeros((IMAX, JMAX))
y = np.zeros((IMAX, JMAX))
phi = np.zeros((IMAX, JMAX))
psi = np.zeros((IMAX, JMAX))

## Grid Initialization by Transfinite grid
# Lower half boundary points
r1 = 0.9  # Ratio in ksi direction in the cut
h = 15.0 * (1.0 - r1) / (1.0 - pow(r1, (IMAX - 1) / 5.0))  # Smallest h in ksi direction in the cut

dh = 0.0
for i in range((IMAX - 1) // 2 + 1):
    if i < (IMAX - 1) // 5:
        x[i][0] = 16.0 - dh
        y[i][0] = 0.0
        x[i][JMAX-1] = 16.0 - dh
        y[i][JMAX-1] = -10.0
        dh += h * pow(r1, float(i))
    elif i < 7 * (IMAX - 1) // 20:
        x[i][0] = 0.5 + 0.5 * math.cos(0.5 * PI * (i - (IMAX - 1) / 5) / (7 * (IMAX - 1) / 20 - (IMAX - 1) / 5))
        y[i][0] = -5.0 * thickness * (0.2948 * pow(x[i][0], 0.5) - 0.126 * x[i][0] - 0.3516 * pow(x[i][0], 2.0) +
                                      0.2843 * pow(x[i][0], 3.0) - 0.1015 * pow(x[i][0], 4.0))
        x[i][JMAX-1] = 1.0 - 15.0 * pow(math.sin((i - (IMAX - 1) / 5) * PI / (2 * ((IMAX - 1) / 2 - (IMAX - 1) / 5))), 1.3)
        y[i][JMAX-1] = -10.0 * (math.cos((i - (IMAX - 1) / 5) * PI / (2 * ((IMAX - 1) / 2 - (IMAX - 1) / 5))))
    else:
        x[i][0] = 0.5 - 0.5 * math.sin(0.5 * PI * (i - (7 * (IMAX - 1) / 20)) / (7 * (IMAX - 1) / 20 - (IMAX - 1) / 5))
        y[i][0] = -5.0 * thickness * (0.2948 * pow(x[i][0], 0.5) - 0.126 * x[i][0] - 0.3516 * pow(x[i][0], 2.0) +
                                      0.2843 * pow(x[i][0], 3.0) - 0.1015 * pow(x[i][0], 4.0))
        x[i][JMAX-1] = 1.0 - 15.0 * pow(math.sin((i - (IMAX - 1) / 5) * PI / (2 * ((IMAX - 1) / 2 - (IMAX - 1) / 5))), 1.3)
        y[i][JMAX-1] = -10.0 * (math.cos((i - (IMAX - 1) / 5) * PI / (2 * ((IMAX - 1) / 2 - (IMAX - 1) / 5))))

dxi = 1.0 / (IMAX - 1)
deta = 1.0 / (JMAX - 1)

# Upper half of the boundary points
for i in range((IMAX - 1) // 2 + 1, IMAX):
    x[i][0] = x[(IMAX - 1) - i][0]
    y[i][0] = -y[(IMAX - 1) - i][0]
    x[i][JMAX-1] = x[(IMAX - 1) - i][JMAX-1]
    y[i][JMAX-1] = -y[(IMAX - 1) - i][JMAX-1]


# Boundary Points at outflow
r2 = 1.1 # Ratio in eta direction at the outflow
h = 10.0 * (1.0 - r2) / (1.0 - pow(r2, (JMAX - 1)))  # Smallest h in eta direction at the outflow

dh = 0.0
for j in range(JMAX):
    x[0][j] = 16.0
    y[0][j] = -dh
    x[IMAX-1][j] = 16.0
    y[IMAX-1][j] = dh
    dh += h * pow(r2, float(j))

# Trans-Finite interpolation for inner points
for j in range(1, JMAX - 1):
    for i in range(1, IMAX - 1):
        x[i][j] = (i / (IMAX - 1)) * x[IMAX-1][j] + ((IMAX - 1 - i) / (IMAX - 1)) * x[0][j] + \
                  (j / (JMAX - 1)) * x[i][JMAX-1] + ((JMAX - 1 - j) / (JMAX - 1)) * x[i][0] - \
                  (i / (IMAX - 1)) * (j / (JMAX - 1)) * x[IMAX-1][JMAX-1] - \
                  (i / (IMAX - 1)) * ((JMAX - 1 - j) / (JMAX - 1)) * x[IMAX-1][0] - \
                  ((IMAX - 1 - i) / (IMAX - 1)) * (j / (JMAX - 1)) * x[0][JMAX-1] - \
                  ((IMAX - 1 - i) / (IMAX - 1)) * ((JMAX - 1 - j) / (JMAX - 1)) * x[0][0]
        
        y[i][j] = (i / (IMAX - 1)) * y[IMAX-1][j] + ((IMAX - 1 - i) / (IMAX - 1)) * y[0][j] + \
                  (j / (JMAX - 1)) * y[i][JMAX-1] + ((JMAX - 1 - j) / (JMAX - 1)) * y[i][0] - \
                  (i / (IMAX - 1)) * (j / (JMAX - 1)) * y[IMAX-1][JMAX-1] - \
                  (i / (IMAX - 1)) * ((JMAX - 1 - j) / (JMAX - 1)) * y[IMAX-1][0] - \
                  ((IMAX - 1 - i) / (IMAX - 1)) * (j / (JMAX - 1)) * y[0][JMAX-1] - \
                  ((IMAX - 1 - i) / (IMAX - 1)) * ((JMAX - 1 - j) / (JMAX - 1)) * y[0][0]

with open("trnsFinGrid.dat", "w") as pt1:
    pt1.write("VARIABLES = \"x\",\"y\"\n")
    pt1.write(f"ZONE T = \"0\" I = {IMAX} J = {JMAX}\n")
    for j in range(JMAX):
        for i in range(IMAX):
            pt1.write(f"{x[i][j]} {y[i][j]}\n")

print("Boundary points initialized, grid generated, and data saved to 'trnsFinGrid.dat'.")


##Optimize the mesh by Elliptic Grid Generation techniques
# Solving Poisson equation using Jacobian iteration
omega = 0.3
residual = 1.0
alf = 0.0
beta = 0.0
gam = 0.0
xtemp = 0.0
ytemp = 0.0
ITR = 0
#a = 0.1
a = 0.005
#aa = 50
aa = 75
#c = 3.6
c = 15
#cc = 8.5
cc = 10.5
dxi = 1.0 / (IMAX - 1)
deta = 1.0 / (JMAX - 1)
xii = 0.5
etai = 0.0
lamb=0.5
epsilon=1e-6
x_1 = 0
y_1 = 0
xm=0
ym=0
#phi and psi for orthognality
def cal_force_func(x,y):
    for j in range(JMAX):
        for i in range(IMAX):
            if j == 0 or j == JMAX-1:
                if i == 0:
                    xxi= (x[i + 1][j] - x[i][j]) / (dxi)
                    yxi = (y[i + 1][j] - y[i][j]) / (dxi)
                    xxi2 = (x[i + 2][j] - 2 * x[i + 1][j] + x[i][j]) / (dxi * dxi)
                    yxi2 = (y[i + 2][j] - 2 * y[i + 1][j] + y[i][j]) / (dxi * dxi)
            
                if i == JMAX-1:
                    xxi= (x[i][j] - x[i - 1][j]) / (dxi)
                    yxi = (y[i][j] - y[i - 1][j]) / (dxi)
                    xxi2 = (x[i - 2][j] - 2 * x[i - 1][j] + x[i][j]) / (dxi * dxi)
                    yxi2 = (y[i - 2][j] - 2 * y[i - 1][j] + y[i][j]) / (dxi * dxi)

                if i > 0 and i < IMAX-1:
                    xxi= (x[i + 1][j] - x[i - 1][j]) / (2.0 * dxi)
                    yxi = (y[i + 1][j] - y[i - 1][j]) / (2.0 * dxi)
                    xxi2 = (x[i + 1][j] - 2 * x[i][j] + x[i - 1][j]) / (dxi * dxi)
                    yxi2 = (y[i + 1][j] - 2 * y[i][j] + y[i - 1][j]) / (dxi * dxi)
                
                gam = (xxi * xxi + yxi * yxi)
            
                if j == 0:
                    xeta = (x[i][j + 1] - x[i][j]) / (deta)
                    yeta = (y[i][j + 1] - y[i][j]) / (deta)

                    xeta1 = -yxi / gam * (-yxi * xeta + xxi * yeta)
                    yeta1 = xxi / gam * (-yxi * xeta + xxi * yeta)

                    x_1 = x[i][j] - xeta1 * deta
                    y_1 = y[i][j] - yeta1 * deta

                    xeta2 = (x[i][j + 1] - 2 * x[i][j] + x_1) / (deta * deta)
                    yeta2 = (y[i][j + 1] - 2 * y[i][j] + y_1) / (deta * deta)
            
                if j == JMAX-1:
                    xeta = (x[i][j - 1] - x[i][j]) / (-deta)
                    yeta = (y[i][j - 1] - y[i][j]) / (-deta)

                    xeta1 = -yxi / gam * (-yxi * xeta + xxi * yeta)
                    yeta1 = xxi / gam * (-yxi * xeta + xxi * yeta)

                    xm = x[i][j] + xeta * deta
                    ym = y[i][j] + yeta * deta

                    xeta2 = (x[i][j - 1] - 2 * x[i][j] + xm) / (deta * deta)
                    yeta2 = (y[i][j - 1] - 2 * y[i][j] + ym) / (deta * deta)

            if i == 0 or i == IMAX-1:

                if j == 0:
                    xeta = (x[i][j + 1] - x[i][j]) / (deta)
                    yeta = (y[i][j + 1] - y[i][j]) / (deta)
                    xeta2 = (x[i][j + 2] - 2 * x[i][j+1] + x[i][j]) / (deta * deta)
                    yeta2 = (y[i][j + 2] - 2 * y[i][j+1] + y[i][j]) / (deta * deta)
            
                if j == JMAX-1:
                    xeta = (x[i][j - 1] - x[i][j]) / (-deta)
                    yeta = (y[i][j - 1] - y[i][j]) / (-deta)
                    xeta2 = (x[i][j - 2] - 2 * x[i][j-1] + x[i][j]) / (deta * deta)
                    yeta2 = (y[i][j - 2] - 2 * y[i][j-1] + y[i][j]) / (deta * deta)

                if j > 0 and j < JMAX-1:
                    xeta = (x[i][j + 1] - x[i][j - 1]) / (2.0 * deta)
                    yeta = (y[i][j + 1] - y[i][j - 1]) / (2.0 * deta)
                    xeta2 = (x[i][j + 1] - 2 * x[i][j] + x[i][j - 1]) / (deta * deta)
                    yeta2 = (y[i][j + 1] - 2 * y[i][j] + y[i][j - 1]) / (deta * deta)
                    
                alf = (xeta * xeta + yeta * yeta)

                if i == 0:
                    xxi = (x[i + 1][j] - x[i][j]) / (dxi)
                    yxi = (y[i + 1][j] - y[i][j]) / (dxi)

                    xxi1 = yeta / alf * (yeta * xxi - xeta * yxi)
                    yxi1 = - xeta / alf * (yeta * xxi - xeta * yxi)

                    x_1 = x[i][j] - xxi1 * dxi
                    y_1 = y[i][j] - yxi1 * dxi

                    xxi2 = (x[i + 1][j] - 2 * x[i][j] + x_1) / (dxi * dxi)
                    yxi2 = (y[i + 1][j] - 2 * y[i][j] + y_1) / (dxi * dxi)
                
                if i == IMAX-1:
                    xxi = (x[i - 1][j] - x[i][j]) / (-dxi)
                    yxi = (y[i - 1][j] - y[i][j]) / (-dxi)

                    xxi1 = yeta / alf * (yeta * xxi - xeta * yxi)
                    yxi1 = - xeta / alf * (yeta * xxi - xeta * yxi)

                    xm = x[i][j] + xxi1 * dxi
                    ym = y[i][j] + yxi1 * dxi

                    xxi2 = (xm - 2 * x[i][j] + x[i-1][j]) / (dxi * dxi)
                    yxi2 = (ym - 2 * y[i][j] + y[i-1][j]) / (dxi * dxi)
                
            
            if xxi != 0 or yxi != 0 :
                phi[i][j] = -(yxi * yxi2 + xxi * xxi2) / (xxi * xxi + yxi * yxi) - (xxi * xeta2 + yxi * yeta2) / (xeta1 * xeta1 + yeta1 * yeta1)
            else:
                phi[i][j]=0
            if xeta != 0 or yeta != 0:
                psi[i][j] = -(yeta * yeta2 + xeta * xeta2) / (xeta * xeta + yeta * yeta) -  (xxi2 * xeta + yxi2 * yeta) / (xxi1 * xxi1 + yxi1 * yxi1)
            else:
                psi[i][j]=0
    # Linearly interpolate internally
    for j in range(1, JMAX-1):
        for i in range(1, IMAX-1):
            psi[i][j] = (i / (IMAX - 1)) * psi[IMAX-1][j] + ((IMAX - 1 - i) / (IMAX - 1)) * psi[0][j]
            #+(j / (JMAX - 1)) * psi[i][JMAX-1] + ((JMAX - 1 - j) / (JMAX - 1)) * psi[i][0]
            phi[i][j] = (j / (JMAX - 1)) * phi[i][JMAX-1] + ((JMAX - 1 - j) / (JMAX - 1)) * phi[i][0]
            #+(i / (IMAX - 1)) * phi[IMAX-1][j] + ((IMAX - 1 - i) / (IMAX - 1)) * phi[0][j] 
    
    return(phi, psi)


while residual > 1e-6 and ITR<100000:  # Convergence Criterion
    residual = 0.0
    ITR += 1
    phi, psi = cal_force_func(x,y)
    #print(phi)
    #psi = np.zeros_like(x)
    #phi = np.zeros_like(x)

    for j in range(1, JMAX - 1):
        for i in range(1, IMAX - 1):
            xeta = (x[i][j + 1] - x[i][j - 1]) / (2.0 * deta)
            xeta2 = (x[i][j + 1] - 2 * x[i][j] + x[i][j - 1]) / (deta * deta)
            yeta = (y[i][j + 1] - y[i][j - 1]) / (2.0 * deta)
            yeta2 = (y[i][j + 1] - 2 * y[i][j] + y[i][j - 1]) / (deta * deta)

            xxi = (x[i + 1][j] - x[i - 1][j]) / (2.0 * dxi)
            xxi2 = (x[i + 1][j] -2 * x[i][j] + x[i - 1][j]) / (dxi * dxi)
            yxi = (y[i + 1][j] - y[i - 1][j]) / (2.0 * dxi)
            yxi2 = (y[i + 1][j] -2 * y[i][j] + y[i - 1][j]) / (dxi * dxi)

            J = xxi * yeta - xeta * yxi
            #print(J)
            J = J if abs(J)<1e6 else 1e6
            xix = 1 / J * yeta
            xiy = -1 / J * xeta
            etax = -1 / J * yxi
            etay = 1 / J * xxi

            alf = (xeta * xeta + yeta * yeta)
            beta = (xxi * xeta + yxi * yeta)
            gam = (xxi * xxi + yxi * yxi)

            #The active terms control the normal spacing
            if abs((i / (IMAX - 1)) - xii) == 0.0:
                PP1 = 0.0
            else:
                PP1 = -a * np.sign((i / (IMAX - 1)) - xii) * math.exp(-c * abs((i / (IMAX - 1)) - xii))

            PP=0
            if abs((j / (JMAX - 1)) - etai) == 0.0:
                QQ1 = 0.0
            else:
                QQ1 = -aa* np.sign((j / (JMAX - 1)) - etai) * math.exp(-cc * abs((j / (JMAX - 1)) - etai))
            
            
            #The active terms control the orthogonality
            #PP2 = phi[i][j] *(xix*xix+xiy*xiy) 
            #QQ2 = psi[i][j] *(etax*etax+etay*etay)
            #QQ2=0
            #print(PP2,QQ2)

            # Dynamic control based on residual
            '''weight=0.3
            PP = (1 - weight) * PP1 + weight * PP2
            QQ = (1 - weight) * QQ1 + weight * QQ2'''
            
            PP = phi[i][j] *(xix*xix+xiy*xiy)+PP1 
            #+PP1
            QQ = psi[i][j] *(etax*etax+etay*etay) + QQ1
            bp = (2.0 * (alf * deta * deta + gam * dxi * dxi)) / pow(dxi * deta, 2.0)
            bw = alf / (dxi * dxi)
            bs = gam / (deta * deta)
            cpx = -beta / (2.0 * dxi * deta) * (x[i + 1][j + 1] + x[i - 1][j - 1] - x[i - 1][j + 1] - x[i + 1][j - 1])
            cpy = -beta / (2.0 * dxi * deta) * (y[i + 1][j + 1] + y[i - 1][j - 1] - y[i - 1][j + 1] - y[i + 1][j - 1])
            
            xtemp = (bw * (x[i + 1][j] + x[i - 1][j]) + bs * (x[i][j + 1] + x[i][j - 1]) + cpx + (J * J) * (xxi * PP + xeta * QQ))/bp
            ytemp = (bw * (y[i + 1][j] + y[i - 1][j]) + bs * (y[i][j + 1] + y[i][j - 1]) + cpy + (J * J) * (yxi * PP + yeta * QQ))/bp

            '''xtemp = pow(dxi * deta, 2.0) / (2.0 * (alf * deta * deta + gam * dxi * dxi)) * \
                    (alf / (dxi * dxi) * (x[i + 1][j] + x[i - 1][j]) +
                     gam / (deta * deta) * (x[i][j + 1] + x[i][j - 1]) -
                     beta / (2.0 * dxi * deta) * (x[i + 1][j + 1] + x[i - 1][j - 1] - x[i - 1][j + 1] - x[i + 1][j - 1]) +
                     (J * J) * (xxi * PP + xeta * QQ))

            ytemp = pow(dxi * deta, 2.0) / (2.0 * (alf * deta * deta + gam * dxi * dxi)) * \
                    (alf / (dxi * dxi) * (y[i + 1][j] + y[i - 1][j]) +
                     gam / (deta * deta) * (y[i][j + 1] + y[i][j - 1]) -
                     beta / (2.0 * dxi * deta) * (y[i + 1][j + 1] + y[i - 1][j - 1] - y[i - 1][j + 1] - y[i + 1][j - 1]) +
                     (J * J) * (yxi * PP + yeta * QQ))'''

            residual += pow((x[i][j] - xtemp), 2.0) + pow((y[i][j] - ytemp), 2.0)

            xtemp = omega * xtemp + (1.0 - omega) * x[i][j]
            ytemp = omega * ytemp + (1.0 - omega) * y[i][j]
            x[i][j] = xtemp
            y[i][j] = ytemp

    residual = math.sqrt(residual)
    print(f"Iteration= {ITR} with residual= {residual}")

    if ITR % 2000 == 0:
        filename = f"ellipGrid_{ITR}.dat"
        with open(filename, "w") as pt2:
            pt2.write("VARIABLES = \"x\",\"y\"\n")
            pt2.write(f"ZONE T = \"0\" I = {IMAX} J = {JMAX}\n")
            for j in range(JMAX):
                for i in range(IMAX):
                    pt2.write(f"{x[i][j]} {y[i][j]}\n")
        print(f"Grid saved to '{filename}' at iteration {ITR}.")

print(f"Converged after {ITR} iterations with residual {residual}")

# Writing the final grid to a file
with open("ellipGrid1.dat", "w") as pt2:
    pt2.write("VARIABLES = \"x\",\"y\"\n")
    pt2.write(f"ZONE T = \"0\" I = {IMAX} J = {JMAX}\n")
    for j in range(JMAX):
        for i in range(IMAX):
            pt2.write(f"{x[i][j]} {y[i][j]}\n")

print("Final grid generated and saved to 'ellipGrid_org.dat'.")
