# Face Recognition - PubFig83

## 1. Informações gerais

Modelo de redes neurais com [TensorFlow](https://www.tensorflow.org/) para
reconhecimento facial, utilizando o dataset
[PubFig83](http://www.briancbecker.com/blog/research/pubfig83-lfw-dataset/)

## 2. Informações sobre o conjunto de dados

O conjunto de imagens para treinamento e testes foi previamente estabelecido,
tendo 12.178 amostras de treinamento e 1.660 amostras para teste. As imagens de
teste estão igualmente divididas entre as classes, sendo 20 imagens para cada
uma das 83 classes.

Todas as imagens possuem o mesmo formato, de 100x100 pixels e 3 camadas de cores
(formato 100 x 100 x 3). Os valores de cores variam entre 0 e 255 (informação de
cor de 8 bits).

As classes não estão balanceadas, possuindo números diferentes de amostras. O
número de imagens em cada classe varia entre 100 e 367.

## 3. Pré-processamento

A primeira operação de preparação dos dados que realizamos foi adicionar uma
coluna com com o nome das classes aos _dataframes_ de treinamento e teste
(carregados a partir dos arquivos [train.csv](./Datasets/train.csv) e
[test.csv](./Datasets/test.csv)). Simplesmente extraímos a informação do nome do
diretório de cada imagem do caminho completo para o arquivo.

Ainda na preparação de dados, criamos um terceiro _dataframe_ para validação do
treinamento. Optamos por seguir a mesma lógica dos dados de teste, escolhendo 20
imagens aleatórias de cada classe para compor o conjunto de validação.

Para balancear o número de amostras de treinamento em cada classe, nossa
primeira abordagem foi realizar o _data augmentation_ de forma a deixar todas as
classes com o mesmo número de amostras que a maior classe. Esta abordagem,
porém, esbarrou em dois problemas. O primeiro foi uma grande quantidade total de
amostras ao final, dificultando as operações sobre o _dataset_ como um todo
(tanto no momento de gerar e salvar o novo conjunto de dados de treinamento
quanto na maior carga no momento de treinar os modelos). O segundo problema era
que as classes com menos imagens chegavam a menos 20% do número de amostras da
maior classe, fazendo com que muitas de suas imagens de treinamento fossem
apenas pequenas variações de uma mesma imagem original.

Por conta disto, optamos primeiramente por realizar o _undersampling_ das
amostras de treinamento, deixando cada classe com no máximo 200 imagens.

O pré-processamento em si foi bastante simples, consistindo essencialmente em:

1. Criação do _dataset_ `X`: carregamento dos dados dos arquivos de imagens utilizando
   [OpenCV](https://opencv.org/);
1. _Feature scaling_: divisão dos dados de cor por `255`, de forma que os dados
   estivessem escalonados entre `0` e `1`.
1. Criação do _dataset_ `y`: utilização do
   [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
   do _sklearn_ para codificar os rótulos em valores numéricos.

Estas operações foram organizadas em uma função `preprocess_dataset()`, de forma
que pudessem ser facilmente aplicadas aos conjuntos de dados de treinamento,
validação e teste.

Finalmente realizamos o _data augmentation_, gerando novas imagens para alcançar
o balanceamento do número de amostras entre todas as classes. As novas imagens
foram geradas a partir de um espelhamento aleatório
([RandomFlip](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomFlip))
na horizontal (aleatoriamente a imagem era espelhada ou não) da imagem original;
e uma rotação aleatória
([RandomRotation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomRotation))
com fator de `0.1` (`[-10% * 2$\pi$, 10% * 2$\pi$]`). Tentamos usar também uma
operação de deslocamento aleatório
([RandomTranslation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomTranslation)),
porém não conseguimos por conta de incompatibilidade com a arquitetura do
computador utilizado.

## 4. Modelos

Para treinamento e validação dos modelos, criamos inicialmente uma função
auxiliar `fit_model()` que recebe como parâmetros o modelo a ser treinado, o
otimizador a ser utilizado na compilação do modelo e os _dataset_ de treinamento
e validação. Nesta função, estabelecemos alguns parâmetros de treinamento que
perpassam os diferentes modelos testados:

- Como função de _loss_, usamos a
  [sparse_categorical_crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/sparse_categorical_crossentropy)
- A métrica utilizada é a
  [acurácia](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy)
- Treinamento do modelo em `100` épocas, porém adotando a estratégia de
  [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
  com paciência de `10` épocas.

A ideia do primeiro modelo é treinar uma arquitetura _do zero_, servindo como
uma espécie de _baseline_ para os demais. A rede é composta de duas repetições
de uma camada convolucional com 32 filtros e kernel de tamanho 4x4, seguida de
uma camada de _max pooling_ com _pool size_ 2x2. Ao final, uma camada _fully
connected_ com 512 neurônios e, na saída, uma camada também _fully connected_ do
tamanho do número de classes e ativada com uma função
[softmax](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax),
de forma a fornecer a classe com maior probabilidade. Como otimizador para
treinamento da rede, utilizamos o
[Stochastic Gradient Descent](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD).
A acurácia deste modelo sobre os dados de validação ficou em torno de 40%.

No segundo modelo, nossa ideia foi utilizar uma arquitetura de rede pré-definida
([ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50))
e, ao final, utilizar a mesma lógica de redes _fully connected_ (neste caso,
usamos 3 camadas com 512 neurônios cada) e uma camada `softmax` na saída. Apesar
de, por padrão, esta rede no Keras ser iniciada com os pesos de treinamento de
dados da [ImageNet](https://image-net.org/), decidimos por retreinar todos os
parâmetros da rede (ou seja, só "aproveitar" sua arquitetura). Desta forma, das
redes que treinamos esta era a que possuía a maior quantidade de parâmetros a
serem aprendidos, quase 41 milhões. Para treinamento desta rede, utilizamos um
otimizador
[NAdam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam).
Sua performance foi consideravelmente melhor que a anterior, chegando a
quase 65% de acurácia nos dados de validação.

No terceiro modelo queríamos testar a técnica de _transfer learning_. Para isto,
criamos uma rede utilizando o modelo
[VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16)
e os pesos de treinamento da ImageNet. Configuramos os parâmetros desta rede
como não treináveis. Na sequência, acoplamos 2 camadas densas com 256 neurônios
cada e a mesma camada `softmax` na saída. O modelo foi treinado com otimizador
[Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) e
apresentou acurácia de validação bem próxima do _baseline_, em torno de 42%.

Para este terceiro modelo, testamos algumas variações nas camadas densas.
Chegamos a testar até 8 camadas com 1024 neurônios, porém não conseguimos
melhores resultados. Pelo contrário, percebemos que um aumento das camadas
densas tendeu a uma piora da acurácia de validação.

Nosso quarto e último modelo desenvolvido utilizou uma rede neural para redução
de dimensionalidade e a classificação ficou a cargo de um algoritmo
classificador linear. Para a primeira parte, utilizamos uma rede
[VGG19](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19)
com os pesos de treinamento da ImageNet. Não realizamos nenhum treinamento
adicional desta rede, apenas utilizamos o preditor para os dados de treinamento
e validação. A saída foi então utilizada como _feature_ de entrada para um
classificador
[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
A acurácia de validação deste modelo ficou pouco abaixo de 50%.

Experimentamos trocar o classificador linear, utilizando um
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
porém os resultados foram muito piores, com acurácia em torno de 23% apenas.

## 5. Testes

Executamos os quatro modelos acima nos dados de teste e as acurácias obtidas
foram muito próximas do que já havíamos verificado durante a validação.

| Modelo           | Acurácia | Tempo para treinamento |
| ---------------- | -------- | ---------------------- |
| Do zero          | 0,42     | 5min 17s               |
| ResNet50         | 0,64     | 1h 59min 49s           |
| VGG16 & ImageNet | 0,40     | 18min 44s              |
| VGG19 + SVC      | 0,49     | 4min 20s               |

## 6. Conclusão

Os modelos desenvolvidos apresentam uma boa acurácia, considerando a
complexidade do problema de classificação com tantas classes. Em especial, o
modelo que utiliza como base a arquitetura da `ResNet50`, com acurácia 30%
melhor que o segundo resultado.

Não conseguimos generalizar uma ideia dos principais fatores ou caminhos para
melhoria de um modelo. Parece ainda, para nós, mais uma questão de tentativas e
erros, em busca de uma arquitetura e conjuntos de parâmetros que apresentem
resultados mais precisos. Considerando nossa limitação de recursos
computacionais e de tempo, não conseguimos testar muitas combinações além das
descritas aqui. Existe ainda uma infinidade de modelos e hiper-parâmetros
possíveis de serem aplicados e, havendo-se condições para tanto, é provável que
melhores resultados possam ser obtidos.

O modelo baseado na `ResNet50` saiu na liderança dos resultados de forma
disparada, porém às custas de um tempo de execução muito superior a todos os
demais. Imaginamos que em um problema com mais dados (em especial imagens
maiores), o treinamento deste modelo ficaria inviabilizado em uma estação de
trabalho doméstica. Mesmo sendo possível a redução de dimensionalidade das
imagens, nossa experiência prática foi de que, para os mesmos modelos, quando
reduzimos as dimensões das imagens (chegamos a testar reduzir as imagens para
64x64 pixels, mantendo os 3 canais de cores), os resultados foram
consideravelmente piores.

Quando consideramos, para além da acurácia, o fator tempo de treinamento, o
modelo de extração de características com `VGG19` e classificação com `SVC` se
destaca. Este obteve o segundo melhor resultado, tendo o menor tempo de
execução. Ficou clara, neste ponto, a vantagem da técnica de _transfer
learning_, uma vez que nenhum tipo de treinamento precisou ser realizada na rede
neural deste modelo.
