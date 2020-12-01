import numpy as np

def prepare_data():
    data_file = open("em_data.txt","r")
    xs = data_file.readlines()
    return xs

def prob_x_given_theta(x, mu, sigma):
    probability = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp((-1/2 * (x - mu)**2)/(sigma**2))
    return probability

def get_distance(a1, a2, b1, b2, c1, c2):
    params = [a1, a2, b1, b2, c1, c2]
    assert all(len(param) == len(params[0]) for param in params)
    sum = 0.0
    for i in range(len(a1)):
        sum += (a1[i] - a2[i])**2 + (b1[i] - b2[i])**2 + (c1[i] - c2[i])**2
    distance = np.sqrt(sum)
    return distance

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

def m_step(xs, w, fixed_sigma):
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
        if (not fixed_sigma):
            for i in range(len(w)):
                sigma_k += w[i][k] * (np.float(xs[i]) - mu_k)**2
            sigma_k = np.sqrt(sigma_k / n_k)
        else:
            sigma_k = fixed_sigma
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

# Function to perform step 2 of the experiment
# After we have learned the parameters, run through the dataset
# And compute the summed log-likelihoods
def sum_likelihoods(mus, sigmas, alphas, xs):
    # Initialize a log liklihood sum variable
    summed_log_x = 0.0

    # Compute the log likihood for each x value
    for x in xs:
        # Compute the probability for each set of parameters
        local_probability = 0.0

        # Loop through the parameter lists and calculate the probability for each set of parameters
        # Then multiply the probability by alpha and sum
        for i in range(0, len(mus)):
            local_probability += (alphas[i] * prob_x_given_theta(np.float(x), mus[i], sigmas[i]))
            #print("For i=" + str(i) + " Added local probability " + str((alphas[i] * prob_x_given_theta(np.float(x), mus[i], sigmas[i]))))

        # Finally take the log of the summed probability
        summed_log_x += np.log(local_probability)
        #print("For xs=" + str(xs) + " Added local probability " + str(np.log(local_probability)))

    return summed_log_x

def sum_likelihoods_nolog(mus, sigmas, alphas, xs):
    # Initialize a log liklihood sum variable
    summed_log_x = 0.0

    # Compute the log likihood for each x value
    for x in xs:
        # Compute the probability for each set of parameters
        local_probability = 0.0

        # Loop through the parameter lists and calculate the probability for each set of parameters
        # Then multiply the probability by alpha and sum
        for i in range(0, len(mus), 1):
            local_probability += (alphas[i] * prob_x_given_theta(np.float(x), mus[i], sigmas[i]))
            #print("For i=" + str(i) + " Added local probability " + str((alphas[i] * prob_x_given_theta(np.float(x), mus[i], sigmas[i]))))

        # Finally take the log of the summed probability
        summed_log_x *= local_probability
        #print("For xs=" + str(xs) + " Added local probability " + str(np.log(local_probability)))

    return summed_log_x


def driver(k, xs, fixed_sigma):
    mus, sigmas, alphas = init(k)
    tol = 0.001
    converged = False
    prev_mus = mus
    prev_sigmas = sigmas
    prev_alphas = alphas
    prev_params = prev_mus + prev_sigmas + prev_alphas
    while (not converged):
        w = e_step(xs, mus, sigmas, alphas)
        mus, sigmas, alphas = m_step(xs, w, fixed_sigma)
        distance = get_distance(prev_mus, mus, prev_sigmas, sigmas, prev_alphas, alphas)
        print("distance from previous: " + str(distance))
        if (distance < tol):
            converged = True
            break
        else:
            prev_mus = mus
            prev_alphas = alphas
            prev_sigmas = sigmas
        print("mus:" + str(mus) + ", sigmas: " + str(sigmas) + ", alphas: " + str(alphas))
    return mus, sigmas, alphas

def run_experiment(fixed_sigma=None):
    xs = prepare_data()
    # STEP 2:
    # Run the  initial experiment for K=1
    print("~~~~~~~~~~~~BEGIN k=1~~~~~~~~~~~~~")
    mus_1, sigmas_1, alphas_1 = driver(1, xs, fixed_sigma)
    print("FINAL MUS:" + str(mus_1) + ", FINAL SIGMAS: " + str(sigmas_1) + ", FINAL ALPHAS: " + str(alphas_1))

    # Compute the log likelihoods using the computed parameters
    log_liklihood_1 = sum_likelihoods(mus_1, sigmas_1, alphas_1, xs)
    print("\n LOG LIKELIHOOD FOR DATASET: " + str(log_liklihood_1))
    print("\n ~~~~~~~~~~END k=1~~~~~~~~~~~~~ \n\n\n\n")

    # Run the initial experiment for K=3
    print("~~~~~~~~~~~~BEGIN k=3~~~~~~~~~~~~~")
    mus_3, sigmas_3, alphas_3 = driver(3, xs, fixed_sigma)
    print("FINAL MUS:" + str(mus_3) + ", FINAL SIGMAS: " + str(sigmas_3) + ", FINAL ALPHAS: " + str(alphas_3))

    # Compute the log likelihoods using the computed parameters
    log_liklihood_3 = sum_likelihoods(mus_3, sigmas_3, alphas_3, xs)
    print("\n LOG LIKELIHOOD FOR DATASET: " + str(log_liklihood_3))
    print("\n ~~~~~~~~~~END k=3~~~~~~~~~~~~~ \n\n\n\n")

    # Run the initial experiment for K=5
    print("~~~~~~~~~~~~BEGIN k=5~~~~~~~~~~~~~")
    mus_5, sigmas_5, alphas_5 = driver(5, xs, fixed_sigma)
    print("FINAL MUS:" + str(mus_5) + ", FINAL SIGMAS: " + str(sigmas_5) + ", FINAL ALPHAS: " + str(alphas_5))

    # Compute the log likelihoods using the computed parameters
    log_liklihood_5 = sum_likelihoods(mus_5, sigmas_5, alphas_5, xs)
    print("\n LOG LIKELIHOOD FOR DATASET: " + str(log_liklihood_5))
    print("\n ~~~~~~~~~~END k=5~~~~~~~~~~~~~ \n\n\n\n")

    return ""

# Call functions here
run_experiment(fixed_sigma=1.0)
run_experiment()
