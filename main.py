import numpy as np


class PerceptronMulticapa:
    def __init__(self, tam_entrada, tam_oculta, tam_salida, learning_rate=0.01):
        self.tam_entrada = tam_entrada
        self.tam_oculta = tam_oculta
        self.tam_salida = tam_salida
        self.learning_rate = learning_rate

        # Inicialización de los pesos con valores aleatorios entre 0 y 1
        self.pesos_entrada_oculta = np.round(np.random.rand(tam_entrada, tam_oculta), 2)
        self.pesos_oculta_salida = np.round(np.random.rand(tam_oculta, tam_salida), 2)

        # Variable para almacenar la salida de la capa oculta
        self.hidden_layer_output = None

    def step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def fwd_propagation(self, x):
        # Aplanar la entrada
        x_flat = x.reshape(-1, self.tam_entrada)

        # Propagación hacia adelante (forward propagation)
        self.hidden_layer_output = self.step_function(np.round(np.dot(x_flat, self.pesos_entrada_oculta), 2))
        output_layer_output = self.softmax(np.dot(self.hidden_layer_output, self.pesos_oculta_salida))
        return output_layer_output

    def backward_propagation(self, x, y):
        # Propagación hacia atrás (backward propagation)
        output = self.fwd_propagation(x)

        # Calcular los errores en la capa de salida y oculta
        output_error = output - y
        hidden_error = np.dot(output_error, self.pesos_oculta_salida.T) * (
                    self.step_function(np.dot(x, self.pesos_entrada_oculta)) * (
                        1 - self.step_function(np.dot(x, self.pesos_entrada_oculta))))

        # Actualizar los pesos
        self.pesos_oculta_salida -= np.dot(self.hidden_layer_output.T, output_error) * self.learning_rate
        self.pesos_entrada_oculta -= np.dot(x.T, hidden_error) * self.learning_rate

    def predict(self, x):
        return np.argmax(self.fwd_propagation(x), axis=1)


def generar_movimiento_lineal():
    x = np.linspace(0, 10, 10) + np.random.normal(0, 0.1, 10)
    y = 2 * x + np.random.normal(0, 1, 10)
    return np.column_stack((x, y))


def generar_movimiento_circular():
    t = np.linspace(0, 2 * np.pi, 10)  # 10 puntos en el intervalo [0, 2π]
    radio = 1.0  # Radio del círculo

    # Coordenadas (x, y) para el movimiento circular
    x = radio * np.cos(t)
    y = radio * np.sin(t)

    # Crea la matriz
    return np.column_stack((x, y))


def generar_movimiento_aleatorio():
    return np.random.rand(10, 2)


def generar_conjunto_datos(movimiento, num_ejemplos):
    return np.array([movimiento() for _ in range(num_ejemplos)])


def generar_conjunto_prueba():
    datos_lineales = generar_conjunto_datos(generar_movimiento_lineal, 30)
    datos_circulares = generar_conjunto_datos(generar_movimiento_circular, 30)
    datos_aleatorios = generar_conjunto_datos(generar_movimiento_aleatorio, 30)

    return datos_lineales, datos_circulares, datos_aleatorios


def train_perceptron(perceptron, data, targets, epochs=100):
    for _ in range(epochs):
        for i in range(len(data)):
            inputs = data[i].reshape(1, -1)
            target = targets[i].reshape(1, -1)
            perceptron.backward_propagation(inputs, target)


# Generar conjunto de entrenamiento
datos_lineales_entrenamiento = generar_conjunto_datos(generar_movimiento_lineal, 30)
datos_circulares_entrenamiento = generar_conjunto_datos(generar_movimiento_circular, 30)
datos_aleatorios_entrenamiento = generar_conjunto_datos(generar_movimiento_aleatorio, 30)

# Crear etiquetas para cada conjunto de datos
etiquetas_lineales = np.array([[1, 0, 0]] * len(datos_lineales_entrenamiento))
etiquetas_circulares = np.array([[0, 1, 0]] * len(datos_circulares_entrenamiento))
etiquetas_aleatorios = np.array([[0, 0, 1]] * len(datos_aleatorios_entrenamiento))

# Entrenar el perceptrón con los datos de entrenamiento
perceptron = PerceptronMulticapa(tam_entrada=20, tam_oculta=5, tam_salida=3)
train_perceptron(perceptron, np.vstack(
    (datos_lineales_entrenamiento, datos_circulares_entrenamiento, datos_aleatorios_entrenamiento)),
                 np.vstack((etiquetas_lineales, etiquetas_circulares, etiquetas_aleatorios)))

# Generar conjunto de prueba
datos_lineales_prueba, datos_circulares_prueba, datos_aleatorios_prueba = generar_conjunto_prueba()

# Clasificar los ejemplos de prueba
predicciones_lineales = perceptron.predict(datos_lineales_prueba)
predicciones_circulares = perceptron.predict(datos_circulares_prueba)
predicciones_aleatorios = perceptron.predict(datos_aleatorios_prueba)

# Calcular la precisión de las predicciones
precision_lineales = np.mean(predicciones_lineales == 0)
precision_circulares = np.mean(predicciones_circulares == 1)
precision_aleatorios = np.mean(predicciones_aleatorios == 2)

print("Precisión para movimientos lineales:", precision_lineales)
print("Precisión para movimientos circulares:", precision_circulares)
print("Precisión para movimientos aleatorios:", precision_aleatorios)
