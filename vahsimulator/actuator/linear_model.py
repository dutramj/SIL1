import numpy as np
import numba


@numba.njit(cache=True)
def linear_model_2nd_order(
    state: np.ndarray, reference: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """
    @brief Calcula a derivada do estado de um modelo dinâmico linear
    de segunda ordem.

    O modelo representa a dinâmica ideal de um atuador de segunda
    ordem, caracterizado por uma frequência natural e uma razão de
    amortecimento.

    A dinâmica é definida pelas seguintes equações diferenciais:

        x' = v

        v' = wn² (r - x) - 2 ζ wn v

    onde:

    - ``x`` é a posição do atuador;
    - ``v`` é a velocidade do atuador;
    - ``r`` é a referência de posição;
    - ``wn`` é a frequência natural;
    - ``ζ`` é a razão de amortecimento.

    O modelo pode ser escrito na forma de espaço de estados como:

        [x']   [      0          1 ] [x]
        [v'] = [-wn²   -2ζwn ] [v] + [wn²] r

    @param state
        Estado atual do atuador, contendo posição e velocidade.

    @param reference
        Referência de posição aplicada ao atuador.

    @param parameters
        Parâmetros dinâmicos do modelo. O vetor deve conter:

        - ``parameters[0]``: frequência natural ``wn`` [rad/s];
        - ``parameters[1]``: razão de amortecimento ``ζ`` [-].

    @return
        Derivada do estado:

        - ``derivative[0]``: velocidade [rad/s];
        - ``derivative[1]``: aceleração [rad/s²].

    @note
        Esta função representa o modelo ideal do atuador, ou seja,
        não inclui não-linearidades como saturação de posição,
        saturação de velocidade, limitação de aceleração, atraso,
        histerese ou outros efeitos não ideais.

    @note
        A função é compilada pelo Numba em modo ``nopython`` e pode
        ser utilizada diretamente por integradores numéricos
        compatíveis com Numba, como o integrador Runge-Kutta de
        quarta ordem utilizado pelo simulador.

    @note
        O estado e a referência são expressos em unidades angulares
        do sistema interno do simulador, tipicamente radianos.

    @see
        linear_model_1st_order
        Modelo linear de primeira ordem, caso disponível.

    @see
        nonlinear_model_factory
        Fábrica utilizada para combinar este modelo com uma pipeline
        de não-linearidades.
    """

    wn = parameters[0]
    zeta = parameters[1]

    position = state[0]
    velocity = state[1]

    command = reference[0]

    error = command - position

    acceleration = wn**2 * error - 2.0 * zeta * wn * velocity

    derivative = np.empty(2, dtype=np.float64)

    derivative[0] = velocity
    derivative[1] = acceleration

    return derivative
