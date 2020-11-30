import numpy as np

def prob_x_given_theta(x, mu, sigma):
    probability = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp((-1/2 * (x - mu)**2)/(sigma**2))
    return probability

def e_step(xs, mus, sigmas, alphas):
    w = []
    for x in xs:
        w_i = []
        for k in range(len(alphas)):
            denominator = 0.0
            for j in range(len(alphas)):
                denominator += prob_x_given_theta(np.float(x), mus[j], sigmas[j]) * alphas[j]
            w_ik = (prob_x_given_theta(np.float(x), mus[k], sigmas[k]) * alphas[k]) / denominator
            w_i.append(w_ik)
        w.append(w_i)
    return w

def m_step(xs, w):
    n = len(xs)
    mus = []
    alphas = []
    sigmas = []
    for k in range(len(w[0])):
        n_k = 0.0
        mu_k = 0.0
        for i in range(len(w)):
            n_k += w[i][k]
            mu_k += w[i][k] * np.float(xs[i])
        alpha_k = n_k / n
        mu_k = mu_k / n_k
        sigma_k = 0.0
        for i in range(len(w)):
            sigma_k += w[i][k] * (np.float(xs[i]) - mu_k)**2
        sigma_k = np.sqrt(sigma_k / n_k)
        mus.append(mu_k)
        alphas.append(alpha_k)
        sigmas.append(sigma_k)
    return mus, sigmas, alphas

def init(k):
    np.random.seed(0)
    mu_init_max = 25.0
    mu_init_min = 5.0
    sigma_init_max = 1.0
    sigma_init_min = 1.0
    mus = []
    sigmas = []
    alphas = []
    alpha_remaining = 1.0
    for i in range(k):
        mus.append(np.random.uniform(mu_init_min, mu_init_max))
        sigmas.append(np.random.uniform(sigma_init_min, sigma_init_max))
        if (i == k - 1):
            alphas.append(alpha_remaining)
        else:
            alpha = np.random.uniform(0.0, alpha_remaining)
            alphas.append(alpha)
            alpha_remaining = alpha_remaining - alpha
    return mus, sigmas, alphas

with open("em_data.txt","r") as data_file:
    k = 2
    xs = data_file.readlines()
    mus, sigmas, alphas = init(k)
    tol = 0.001
    converged = False
    prev_mus = mus
    prev_sigmas = sigmas
    prev_alphas = alphas
    prev_params = prev_mus + prev_sigmas + prev_alphas
    while (not converged):
        w = e_step(xs, mus, sigmas, alphas)
        mus, sigmas, alphas = m_step(xs, w)
        converged = True
        for k in range(k + 1):
            mu_diff = abs(prev_mus[k] - mus[k])
            sigma_diff = abs(prev_sigmas[k] - sigmas[k])
            alpha_diff = abs(prev_alphas[k] - alphas[k])
            print(mu_diff, sigma_diff, alpha_diff)
            if (mu_diff > tol or sigma_diff > tol or alpha_diff > tol):
                converged = False
                break
        if converged:
            break
        else:
            prev_mus = mus
            prev_alphas = alphas
            prev_sigmas = sigmas
        print("mus:" + str(mus) + ", sigmas: " + str(sigmas) + ", alphas: " + str(alphas))
