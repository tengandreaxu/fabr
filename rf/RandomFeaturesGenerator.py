import numpy as np
from typing import Optional, Tuple

from rf.RandomFeaturesType import RandomFeaturesNames, RandomFeaturesType
import logging

logging.basicConfig(level=logging.INFO)


class RandomFeaturesGenerator:
    """
    This class manages creation of random features

    #########################################
    The list of possible distributions is:
    #########################################
    beta(a, b[, size])	Draw samples from a Beta distribution.
    binomial(n, p[, size])	Draw samples from a binomial distribution.
    chisquare(df[, size])	Draw samples from a chi-square distribution.
    dirichlet(alpha[, size])	Draw samples from the Dirichlet distribution.
    exponential([scale, size])	Draw samples from an exponential distribution.
    f(dfnum, dfden[, size])	Draw samples from an F distribution.
    gamma(shape[, scale, size])	Draw samples from a Gamma distribution.
    geometric(p[, size])	Draw samples from the geometric distribution.
    gumbel([loc, scale, size])	Draw samples from a Gumbel distribution.
    hypergeometric(ngood, nbad, nsample[, size])	Draw samples from a Hypergeometric distribution.
    laplace([loc, scale, size])	Draw samples from the Laplace or double
    exponential distribution with specified location (or mean) and scale (decay).
    logistic([loc, scale, size])	Draw samples from a logistic distribution.
    lognormal([mean, sigma, size])	Draw samples from a log-normal distribution.
    logseries(p[, size])	Draw samples from a logarithmic series distribution.
    multinomial(n, pvals[, size])	Draw samples from a multinomial distribution.
    multivariate_normal(mean, cov[, size, …)	Draw random samples from a multivariate normal distribution.
    negative_binomial(n, p[, size])	Draw samples from a negative binomial distribution.
    noncentral_chisquare(df, nonc[, size])	Draw samples from a noncentral chi-square distribution.
    noncentral_f(dfnum, dfden, nonc[, size])	Draw samples from the noncentral F distribution.
    normal([loc, scale, size])	Draw random samples from a normal (Gaussian) distribution.
    pareto(a[, size])	Draw samples from a Pareto II or Lomax distribution with specified shape.
    poisson([lam, size])	Draw samples from a Poisson distribution.
    power(a[, size])	Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
    rayleigh([scale, size])	Draw samples from a Rayleigh distribution.
    standard_cauchy([size])	Draw samples from a standard Cauchy distribution with mode = 0.
    standard_exponential([size])	Draw samples from the standard exponential distribution.
    standard_gamma(shape[, size])	Draw samples from a standard Gamma distribution.
    standard_normal([size])	Draw samples from a standard Normal distribution (mean=0, stdev=1).
    standard_t(df[, size])	Draw samples from a standard Student’s t distribution with df degrees of freedom.
    triangular(left, mode, right[, size])	Draw samples from the triangular distribution over the interval [left, right].
    uniform([low, high, size])	Draw samples from a uniform distribution.
    vonmises(mu, kappa[, size])	Draw samples from a von Mises distribution.
    wald(mean, scale[, size])	Draw samples from a Wald, or inverse Gaussian, distribution.
    weibull(a[, size])	Draw samples from a Weibull distribution.
    zipf(a[, size])	Draw samples from a Zipf distribution.

    ####################################################################################################################
    activation functions
    ####################################################################################################################
    cos, sin, exp, arctan (sigmoid style!), tanh,
    ReLu (careful, relu is implemented separately through the multiplication method (x * (x > 0)) which is the fastest
    Elu (x * (x > 0)) + alpha * (exp(x) - 1) * (x < 0)
    and SoftPlus = log(1+exp(x))

    """

    # the next shows the number of parameters defining the distribution
    distribution_requirements = {
        "beta": 2,
        "binomial": 2,
        "chisquare": 0,
        "dirichlet": 1,
        "exponential": 0,
        "f": 2,
        "gamma": 1,
        "geometric": 1,
        "gumbel": 2,
        "hypergeometric": 2,
        "laplace": 2,
        "logistic": 2,
        "lognormal": 2,
        "logseries": 1,
        "multinomial": 2,
        "multivariate_normal": 2,
        "negative_binomial": 2,
        "noncentral_chisquare": 2,
        "noncentral_f": 3,
        "normal": 2,
        "pareto": 1,
        "poisson": 1,
        "power": 1,
        "rayleigh": 1,
        "standard_cauchy": 0,
        "standard_exponential": 0,
        "standard_gamma": 1,
        "standard_normal": 0,
        "standard_t": 1,
        "triangular": 3,
        "uniform": 2,
        "vonmises": 2,
        "wald": 2,
        "weibull": 1,
        "zipf": 1,
        "gaussian_mixture": 0,
    }

    permitted_activation_functions = [
        "cos",
        "sin",
        "exp",
        "arctan",
        "tanh",
        "ReLu",
        "Elu",
        "SoftPlus",
        "cos_and_sin",
    ]

    def __init__(self):
        pass

    @staticmethod
    def check_distribution_requirements(
        distribution: str, distribution_parameters: list
    ):

        if distribution == "gaussian_mixture":
            return
        if distribution not in RandomFeaturesGenerator.distribution_requirements:
            raise Exception(
                f"{distribution} is not permitted. If you need it, do not be lazy and update the class"
            )
        elif (
            len(distribution_parameters)
            != RandomFeaturesGenerator.distribution_requirements[distribution]
        ):
            raise Exception(
                f"{distribution} requires {RandomFeaturesGenerator.distribution_requirements[distribution]} parameters"
            )

    @staticmethod
    def get_a_specific_gaussian_mixture_random_features_specification(
        seed: int,
        gamma: list,
        activation: str,
        number_features_in_subset: int,
        increment_seed: Optional[int] = 0,
    ) -> Tuple[int, dict]:
        """returns specification to generate random features"""
        spec = (
            int((seed + 1) * 1e3) + increment_seed,
            {
                "distribution": "gaussian_mixture",
                "distribution_parameters": gamma,
                "activation": activation,
                "number_features": number_features_in_subset,
                "bias_distribution": None,
                "bias_distribution_parameters": [
                    -np.pi,
                    np.pi,
                ],
            },
        )
        return spec

    @staticmethod
    def generate_random_features(
        type: RandomFeaturesType,
        number_features_in_subset: int,
        features: np.ndarray,
        increment_seed: int,
    ) -> np.ndarray:
        """
        Generates Random Features:

        1. Using Random Binning, from Rahimi 2007
        2. Using Random Neurons
        """

        if type.name == RandomFeaturesNames.binning:

            random_features = RandomFeaturesGenerator.generate_random_binning_features(
                features,
                distribution=type.distribution,
                distribution_parameters=type.distribution_parameters,
                random_seed=increment_seed,
                number_features=number_features_in_subset,
            )

        elif type.name == RandomFeaturesNames.neurons:
            random_features = RandomFeaturesGenerator.generate_random_neuron_features(
                features,
                random_seed=increment_seed,
                distribution=type.distribution,
                distribution_parameters=type.distribution_parameters,
                activation=type.activation,
                number_features=number_features_in_subset,
                bias_distribution=type.bias_distribution,
                bias_distribution_parameters=type.bias_distribution_parameters,
            )
        else:
            raise Exception(
                f"{type.name} random feature generation not yet implemented"
            )

        return random_features

    @staticmethod
    def generate_random_features_from_list(
        seed: int,
        gamma: list,
        activation: str,
        number_features_in_subset: int,
        features: np.ndarray,
        increment_seed: int,
        random_cnn: bool = False,
        cnn_bias: Optional[bool] = False,
    ) -> np.ndarray:
        """
        given a list of different specifications, generate random features for each of them
        :param list_of_specs:
        :return:
        """
        # Generate deep random features

        list_of_specs = RandomFeaturesGenerator.get_a_specific_gaussian_mixture_random_features_specification(
            seed=seed,
            gamma=gamma,
            activation=activation,
            number_features_in_subset=number_features_in_subset,
            increment_seed=increment_seed,
        )

        if random_cnn:

            simple_convolution = RandomCNN(seed=seed, bias=cnn_bias)
            random_features = simple_convolution(features).detach().numpy()
            return random_features

        else:
            random_features = []
            if list_of_specs[0] == "mix":
                list_of_specs = list_of_specs[1]
            for spec in list_of_specs:
                # spec[0] is just an index to initialize the np.random.seed
                if len(spec[1]) == 4:
                    random_features.append(
                        RandomFeaturesGenerator.generate_random_binning_features(
                            spec[0], **spec[1]
                        )
                    )
                elif len(spec[1]) == 6:

                    random_features.append(
                        (
                            RandomFeaturesGenerator.generate_random_neuron_features(
                                features,
                                spec[0],
                                **spec[1],
                            )
                        )
                    )

            if len(list_of_specs) > 1:
                merged_features = np.concatenate(random_features, axis=1)
                perm = np.random.permutation(merged_features.shape[1])
                merged_features = merged_features[:, perm]
                return merged_features

            else:
                random_features = random_features[0]
                return random_features

    @staticmethod
    def apply_activation_to_multiplied_signals(
        multiplied_signals, activation: str
    ) -> np.ndarray:
        """
        this method takes as input signaled already multipled by some weights + cosntant: w*x+b
        and returns act(w*x+b)
        :rtype: object
        """

        if activation in ["cos", "sin", "exp", "arctan", "tanh"]:
            final_random_features = getattr(np, activation)(multiplied_signals)
        elif activation == "cos_and_sin":
            final_random_features = np.concatenate(
                [np.cos(multiplied_signals), np.sin(multiplied_signals)], axis=0
            )

        elif isinstance(activation, str) and activation.lower() == "relu":
            final_random_features = multiplied_signals * (multiplied_signals > 0)
        elif isinstance(activation, list) and activation[0].lower() == "elu":
            final_random_features = (
                multiplied_signals * (multiplied_signals > 0)
            ) + activation[1] * (np.exp(multiplied_signals) - 1) * (
                multiplied_signals < 0
            )
        elif isinstance(activation, list) and activation[0].lower() == "leakyrelu":
            final_random_features = (
                multiplied_signals * (multiplied_signals > 0)
            ) + activation[1] * multiplied_signals * (multiplied_signals < 0)
        elif activation.lower() == "softplus":
            final_random_features = np.log(1 + np.exp(multiplied_signals))
        elif activation.lower() == "linear":

            final_random_features = multiplied_signals
        else:
            raise Exception(f"activation function={activation} is not yet supported")
        return final_random_features

    @staticmethod
    def add_bias(
        multiplied_signals,
        bias_distribution,
        bias_distribution_parameters,
        seed=0,
    ):
        """
        Careful, multiplied signals are assumed to be P \times n where P is the sumber of signals
        and n the number of observations
        Parameters
        ----------
        multiplied_signals :
        bias_distribution :
        bias_distribution_parameters :
        Returns
        -------
        """
        np.random.seed(seed)
        number_features = multiplied_signals.shape[0]
        random_bias = getattr(np.random, bias_distribution)(
            *bias_distribution_parameters, [number_features, 1]
        )
        # here we are using numpy broadcasting to add the same bias every time period
        multiplied_signals += random_bias
        return multiplied_signals

    @staticmethod
    def generate_random_neuron_features(
        features: np.ndarray,
        random_seed: int,
        distribution: str,
        distribution_parameters: list,
        activation: str,
        number_features: int,
        bias_distribution=None,
        bias_distribution_parameters=None,
    ) -> np.ndarray:
        """
        this function builds random neuron features f(w'S+bias) where w is
        a vector of random weights and f is an activation function, and bias is a random bias
        :param distribution_parameters:
        :param distribution:
        :param activation:
        :param number_features:
        :param bias_distribution:
        :param index: random seed
        :return:
        """
        np.random.seed(random_seed)
        signals = features

        RandomFeaturesGenerator.check_distribution_requirements(
            distribution, distribution_parameters
        )
        if bias_distribution:
            RandomFeaturesGenerator.check_distribution_requirements(
                bias_distribution,
                bias_distribution_parameters,
            )

        number_signals = signals.shape[1]
        size = [number_signals, number_features]
        if activation == "cos_and_sin":
            size = [number_signals, int(number_features / 2)]

        # first we initialize the random seed

        # X = np.random.normal(0, a) means X is distributed as Normal(0, a^2).  (a=standard deviation)
        # This is an important property of Gaussian distributions: multiplying by a constant keeps is Gaussian,
        # just scales the standard deviation
        if distribution != "gaussian_mixture":

            random_vectors = getattr(np.random, distribution)(
                *distribution_parameters, size
            )
        else:

            random_vectors = getattr(np.random, "standard_normal")(size)

            gamma_values = distribution_parameters
            minimal_gamma = gamma_values[0]
            maximal_gamma = gamma_values[1]
            if minimal_gamma != 1 and maximal_gamma != 1:
                all_gamma_values = np.random.uniform(
                    minimal_gamma, maximal_gamma, [1, size[1]]
                )
                # now we use numpy broadcasting to do elemen-wise multiplication.
                # This is an expensive operation when random_vectors shape is BIG
                # Hence, avoid when gamma = [1,1]
                random_vectors = random_vectors * all_gamma_values
        # w'x, where w is our random vector
        multiplied_signals = np.matmul(random_vectors.T, signals.T)
        if bias_distribution:
            multiplied_signals = RandomFeaturesGenerator.add_bias(
                multiplied_signals, bias_distribution, bias_distribution_parameters
            )
        final_random_features = (
            RandomFeaturesGenerator.apply_activation_to_multiplied_signals(
                multiplied_signals, activation
            )
        )

        return final_random_features.T

    @staticmethod
    def generate_random_binning_features(
        signals: np.ndarray,
        distribution: str,
        distribution_parameters: list,
        number_features: int,
        random_rotation=False,
        random_seed=0,
    ):
        """
        :param random_seed:
        :param random_rotation:
        :param distribution_parameters:
        :param distribution:
        :param distribution_requirements:
        :param signals:
        :param number_features:
        :return:
        """

        RandomFeaturesGenerator.check_distribution_requirements(
            distribution, distribution_parameters
        )

        number_signals = signals.shape[1]
        size = [number_signals, number_features]
        np.random.seed(random_seed)
        if random_rotation:
            rotate = np.random.randn(signals.shape[1], signals.shape[1])
            tmp = np.matmul(rotate, rotate.T)
            _, eigvec = np.linalg.eigh(tmp)
            # now, eigenvectors give a random rotation
            signals_rotated = np.matmul(eigvec.T, signals.T).T
        else:
            signals_rotated = signals.copy()
        delta = getattr(np.random, distribution)(*distribution_parameters, size)
        delta = delta * (np.abs(delta) > (10 ** (-10))) + (
            np.abs(delta) < (10 ** (-10))
        ) * (
            10 ** (-10)
        )  # clip
        u_ = np.random.uniform(0, 1, [number_signals, number_features]) * delta
        subtracted = signals_rotated.reshape(
            signals.shape[0], 1, number_signals
        ) - u_.reshape(1, number_features, number_signals)
        subtracted_and_divided = subtracted / delta.reshape(
            1, number_features, number_signals
        )

        binned_signals = np.floor(subtracted_and_divided).reshape(
            [signals.shape[0], signals.shape[1] * number_features]
        )
        return binned_signals


if __name__ == "__main__":
    X_train = np.random.normal(size=(100, 100))

    distribution = "gaussian_mixture"
    gamma = [1.0, 1.0]
    activation = "relu"

    breakpoint()
