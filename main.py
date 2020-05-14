from network import Network

KEYBOARD = [['', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', ''],
            ['', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ''],
            ['', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '']]

# Criação da rede
weights = [[-1, 2, 1], [1.5, 1, -1], [-2, -1, 1]]

net = Network(2, 0.5, 0.5, 0.2, weights=weights)


# Função para definir se letras estão lado a lado no teclado
def is_lateral(fchar, schar):
    for line in KEYBOARD:
        for pos, char in enumerate(line):
            if char == fchar:
                if schar == line[pos - 1] or schar == line[pos + 1]:
                    return 1
    return 0


# Tratamento das entradas
def input_words(words, network):
    first_length = len(words[0])

    if first_length != len(words[1]):
        raise Exception('Palavras devem ter o mesmo tamanho!')

    first_input = (first_length - 1) / first_length

    for pos, char in enumerate(words[0]):
        if char != words[1][pos]:
            second_input = is_lateral(char.upper(), words[1][pos].upper())

    print(f'Teste: {words[0]} vs {words[1]}')
    print(f'Input 1: {first_input}, Input 2: {second_input} = Output: {network.run([first_input, second_input])}')


# Treinamento
training_data = [[1, 1, 1.00],
                 [1, 0, 0.25],
                 [0, 0, 0.00],
                 [0, 1, 0.75]]

net.train(training_data)

# Teste 1: soja vs soka. Entradas resultantes 0,75 e 1.
input_words(['soja', 'soka'], net)

# Teste 2: soja vs soda
input_words(['soja', 'soda'], net)

# Teste 3: pneumoultramicroscopicossilicovulcanoconiotico vs pneumoultramicroscopicossilicovulcanoconioticp
input_words(['pneumoultramicroscopicossilicovulcanoconiotico', 'pneumoultramicroscopicossilicovulcanoconioticp'], net)

# Teste 4: james vs jimes
input_words(['james', 'jsmes'], net)
