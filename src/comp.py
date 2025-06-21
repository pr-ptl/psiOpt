#------------------------------------------------------------------------------
#   OPTIMIZER COMPARISON
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

#------------------------------------------------------------------------------

def grad_x(f, dx):
    df_dx = np.zeros_like(f)
    df_dx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
    df_dx[0, :] = (f[1, :] - f[0, :]) / dx
    df_dx[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return df_dx

def grad_y(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
    df_dy[:, 0] = (f[:, 1] - f[:, 0]) / dy
    df_dy[:, -1] = (f[:, -1] - f[:, -2]) / dy
    return df_dy

#------------------------------------------------------------------------------

def cmpt_gradanalytic(psi, ut, vt, dx, dy):

    uc = grad_y(psi, dy)
    vc = -grad_x(psi, dx)
    
    uerr = uc - ut
    verr = vc - vt
    
    Np = psi.shape[0] * psi.shape[1]
    au = uerr / Np
    av = verr / Np
    
    grad_u = -grad_y(au, dy)
    grad_v = grad_x(av, dx)
    
    grad = grad_u + grad_v
    
    return grad

def bc(psi):
    """Enforce zero boundary conditions"""
    psi_bc = psi.copy()
    psi_bc[0, :] = 0   # bottom boundary
    psi_bc[-1, :] = 0  # top boundary  
    psi_bc[:, 0] = 0   # left boundary
    psi_bc[:, -1] = 0  # right boundary
    return psi_bc

def zero_bcgrad(grad):
    """Zero out gradients at boundaries"""
    grad_bc = grad.copy()
    grad_bc[0, :] = 0   # bottom boundary
    grad_bc[-1, :] = 0  # top boundary  
    grad_bc[:, 0] = 0   # left boundary
    grad_bc[:, -1] = 0  # right boundary
    return grad_bc

#------------------------------------------------------------------------------
# OPTIMIZERS

def adam_optimizer(psi_init, ut, vt, dx, dy, 
                   max_iter=5000, tol=1e-12, lr_init=1e-3,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer"""
    psi = psi_init.copy()
    psi = bc(psi)
    
    m = np.zeros_like(psi)  # First moment estimate
    v = np.zeros_like(psi)  # Second moment estimate
    t = 0  # Time step
    
    ls = []
    
    for itr in range(max_iter):
        t += 1
        
        uc = grad_y(psi, dy)
        vc = -grad_x(psi, dx)
        uerr = uc - ut
        verr = vc - vt
        loss = 0.5 * np.mean(uerr**2 + verr**2)
        ls.append(loss)
        
        # Check for convergence
        if loss < tol:
            break
            
        grad = cmpt_gradanalytic(psi, ut, vt, dx, dy)
        grad = zero_bcgrad(grad)
        
        m = beta1 * m + (1 - beta1) * grad
        
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**t)
        
        v_hat = v / (1 - beta2**t)
        
        upd = lr_init * m_hat / (np.sqrt(v_hat) + eps)
        psi = psi - upd
        psi = bc(psi)
    
    return psi, np.array(ls)

def rmsprop_optimizer(psi_init, ut, vt, dx, dy, 
                      max_iter=5000, tol=1e-12, lr_init=1e-3,
                      beta=0.9, eps=1e-8):
    """RMSprop optimizer"""
    psi = psi_init.copy()
    psi = bc(psi)
    
    # RMSprop parameters
    v = np.zeros_like(psi)
    ls = []
    
    for itr in range(max_iter):
        
        uc = grad_y(psi, dy)
        vc = -grad_x(psi, dx)
        uerr = uc - ut
        verr = vc - vt
        loss = 0.5 * np.mean(uerr**2 + verr**2)
        ls.append(loss)
        
        # Check for convergence
        if loss < tol:
            break
            
        grad = cmpt_gradanalytic(psi, ut, vt, dx, dy)
        grad = zero_bcgrad(grad)
        
        v = beta * v + (1 - beta) * grad**2
        
        upd = lr_init * grad / (np.sqrt(v) + eps)
        psi = psi - upd
        psi = bc(psi)
    
    return psi, np.array(ls)

def SGD_optimizer(psi_init, ut, vt, dx, dy, 
                              max_iter=5000, tol=1e-12, lr_init=1e-3):
    """Simple gradient descent optimizer"""
    psi = psi_init.copy()
    psi = bc(psi)
    
    ls = []
    
    for itr in range(max_iter):
        
        uc = grad_y(psi, dy)
        vc = -grad_x(psi, dx)
        uerr = uc - ut
        verr = vc - vt
        loss = 0.5 * np.mean(uerr**2 + verr**2)
        ls.append(loss)
        
        # Check for convergence
        if loss < tol:
            break
            
        grad = cmpt_gradanalytic(psi, ut, vt, dx, dy)
        grad = zero_bcgrad(grad)
        
        upd = lr_init * grad
        psi = psi - upd
        psi = bc(psi)
    
    return psi, np.array(ls)

#------------------------------------------------------------------------------
# PROBLEM SETUP
#------------------------------------------------------------------------------

def setup_test(N=100):
    """Set up a test problem with boundary-consistent streamfunction"""
    # Domain setup
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    
    pi = np.pi
    
    psit = np.sin(pi * X) * np.sin(pi * Y)
    ut = grad_y(psit, dy)
    vt = -grad_x(psit, dx)
    
    return X, Y, psit, ut, vt, dx, dy

#------------------------------------------------------------------------------

def run_comp(N=100, max_iter=5000, tol=1e-12):
    
    print("="*80)
    print("OPTIMIZER COMPARISON")
    print("="*80)
    
    # Problem setup
    X, Y, psit, ut, vt, dx, dy = setup_test(N)
    
    print(f"Problem setup:")
    print(f"Grid size: {N}x{N}")
    print(f"Domain: [0,1] x [0,1]")
    print(f"Max iterations: {max_iter}")
    print(f"Tolerance: {tol:.0e}")
    
    # Initial guess
    np.random.seed(42)
    psi_init = np.zeros((N, N))
    psi_init = bc(psi_init)
    
    # Store results
    results = {}
    
    # Adam
    print(f"\n...Running Adam...")
    sta = time.time()
    psi_adam, ls_adam = adam_optimizer(
        psi_init, ut, vt, dx, dy,
        max_iter=max_iter, tol=tol, lr_init=1e-3
    )
    adam_time = time.time() - sta
    
    # RMSprop
    print(f"\n...Running RMSprop...")
    stR = time.time()
    psi_rmsprop, ls_rmsprop = rmsprop_optimizer(
        psi_init, ut, vt, dx, dy,
        max_iter=max_iter, tol=tol, lr_init=1e-3
    )
    rmsprop_time = time.time() - stR
    
    # SGD
    print(f"\n...Running Gradient Descent...")
    stsgd = time.time()
    psi_gd, ls_gd = SGD_optimizer(
        psi_init, ut, vt, dx, dy,
        max_iter=max_iter, tol=tol, lr_init=1e-3
    )
    gd_time = time.time() - stsgd
    
    # Store results
    results = {
        'Adam': {
            'psi': psi_adam,
            'losses': ls_adam,
            'time': adam_time,
            'iterations': len(ls_adam),
            'final_loss': ls_adam[-1],
        },
        'RMSprop': {
            'psi': psi_rmsprop,
            'losses': ls_rmsprop,
            'time': rmsprop_time,
            'iterations': len(ls_rmsprop),
            'final_loss': ls_rmsprop[-1],
        },
        'Gradient Descent': {
            'psi': psi_gd,
            'losses': ls_gd,
            'time': gd_time,
            'iterations': len(ls_gd),
            'final_loss': ls_gd[-1],
        }
    }
    
    return results, psit, X, Y

def plot_comparison(results, psit, X, Y):
    
    print(f"\nERROR ANALYSIS:")
    for method, data in results.items():
        err = psit - data['psi']
        max_err = np.max(np.abs(err))
        rms_err = np.sqrt(np.mean(err**2))
        print(f"{method:<15} Max Error: {max_err:.2e}, RMS Error: {rms_err:.2e}")
    
    cplots(results, psit, X, Y)

def cplots(results, psit, X, Y):

    plt.figure(figsize=(12, 8))
    
    colors = {'Adam': 'blue', 'RMSprop': 'red', 'Gradient Descent': 'green'}
    
    for method, data in results.items():
        plt.semilogy(data['losses'], color=colors[method], linewidth=2, 
                    label=f"{method} (final: {data['final_loss']:.2e})")
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Optimizer Convergence Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('opt_convergence.jpg', dpi=400, bbox_inches='tight')
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    cs0 = axes[0,0].contourf(X, Y, psit, levels=20, cmap='viridis')
    axes[0,0].contour(X, Y, psit, levels=10, colors='k', linewidths=0.5)
    axes[0,0].set_title('True Solution', fontsize=14)
    axes[0,0].set_aspect('equal')
    plt.colorbar(cs0, ax=axes[0,0])
    
    plot_idx = [(0,1), (1,0), (1,1)]
    for i, (method, data) in enumerate(results.items()):
        row, col = plot_idx[i]
        cs = axes[row,col].contourf(X, Y, data['psi'], levels=20, cmap='viridis')
        axes[row,col].contour(X, Y, data['psi'], levels=10, colors='k', linewidths=0.5)
        axes[row,col].set_title(f'{method} Solution', fontsize=14)
        axes[row,col].set_aspect('equal')
        plt.colorbar(cs, ax=axes[row,col])
    
    plt.tight_layout()
    plt.savefig('optimizer_solutions.jpg', dpi=400, bbox_inches='tight')
    plt.show()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (method, data) in enumerate(results.items()):
        err = psit - data['psi']
        max_err = np.max(np.abs(err))
        
        cs = axes[i].contourf(X, Y, err, levels=20, cmap='RdBu_r', 
                             vmin=-max_err, vmax=max_err)
        axes[i].contour(X, Y, err, levels=8, colors='k', linewidths=0.5)
        axes[i].set_title(f'{method} Error\n(max = {max_err:.2e})', fontsize=12)
        axes[i].set_aspect('equal')
        plt.colorbar(cs, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('opt_errors.jpg', dpi=400, bbox_inches='tight')
    plt.show()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = list(results.keys())
    final_losses = [results[m]['final_loss'] for m in methods]
    iterations = [results[m]['iterations'] for m in methods]
    times = [results[m]['time'] for m in methods]
    
    bars1 = ax1.bar(methods, final_losses, color=[colors[m] for m in methods])
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Final Loss Comparison')
    ax1.set_yscale('log')
    for i, v in enumerate(final_losses):
        ax1.text(i, v*1.1, f'{v:.1e}', ha='center', va='bottom', fontsize=10)
    
    bars2 = ax2.bar(methods, iterations, color=[colors[m] for m in methods])
    ax2.set_ylabel('Iterations to Convergence')
    ax2.set_title('Convergence Speed')
    for i, v in enumerate(iterations):
        ax2.text(i, v+max(iterations)*0.01, str(v), ha='center', va='bottom', fontsize=10)
    
    bars3 = ax3.bar(methods, times, color=[colors[m] for m in methods])
    ax3.set_ylabel(r'Time $[s]$')
    ax3.set_title('Computational Time')
    for i, v in enumerate(times):
        ax3.text(i, v+max(times)*0.01, f'{v:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('opt_metrics.jpg', dpi=400, bbox_inches='tight')
    plt.show()

#------------------------------------------------------------------------------

def main():
    
    results, psit, X, Y = run_comp(N=100, max_iter=5000, tol=1e-12)
    
    plot_comparison(results, psit, X, Y)
    
    return results

if __name__ == "__main__":
    results = main()
