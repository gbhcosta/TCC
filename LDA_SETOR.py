# Importando bibliotecas
import pandas as pd
import numpy as np
import re, nltk, gensim
import seaborn as sns
import matplotlib.pyplot as plt 

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# Lendo o arquivo
df = pd.read_csv('Dados_Setor.csv', sep =';')

# Quantidadde de documentos
print("Temos %d documentos." %len(df))

# Retirando linhas indesejadas do DataFrame
df = df[df["ID"] != -1]
print("Temos %d documentos." %len(df))

# Removendo substituição Reclame Aqui
print('Quantidade de documentos contendo texto editado pelo Reclame Aqui:', len(df[df['reclamacao_completa'].str.contains('\[Editado pelo Reclame Aqui\]')]))
df['reclamacao_completa'] = df['reclamacao_completa'].str.replace('\[Editado pelo Reclame Aqui\]','')
print('Quantidade de documentos contendo texto editado pelo Reclame Aqui:', len(df[df['reclamacao_completa'].str.contains('\[Editado pelo Reclame Aqui\]')]))

#   Método que executa o processo de tokenização, transformando todos os tokens em letra minúscula e removendo acentuação e pontuação
def frases_para_palavras(frases):
  for frase in frases:
      yield(gensim.utils.simple_preprocess(str(frase), deacc=True)) # deacc=True remove acentuação e pontuação

# Colocando os tokens em lista para uso posterior
data_words = list(frases_para_palavras(df['reclamacao_completa']))

nltk.download('stopwords')
# Indica que as stopwords são em português
stopwords = nltk.corpus.stopwords.words('portuguese')
# Adicionando palavras ao corpus de stopwords 
new_stopwords = ["nao", "uber", "eats", "ubereats", "ubereat","rappi", "rapi", "ifood", "ifod"]
stopwords.extend(new_stopwords)

# Método para remover as palavras irrelevantes
def removeStops(texts, stopwords):
    texts_out = []
    for sent in texts:
        texts_out.append(" ".join([token for token in sent if token not in stopwords]))
    return texts_out 
# Execução do método
data_without_stops = removeStops(data_words, stopwords)

vectorizer = CountVectorizer(analyzer='word',
                             min_df=5,                         # Define o numero mínimo de ocorrência do token
                             max_df=0.5,                       # Estabelece a porcentagem máxima de ocorrência da palavra nos documentos
                             lowercase=True,                   # Converte todas as palavras apra minúsculas
                             token_pattern='[a-zA-Z0-9]{3,}',  # Padrâo e palavras, que tenham pelo menos 3 letras
                            )
data_vectorized = vectorizer.fit_transform(data_without_stops)
#print (data_vectorized)
'''
Basta verificar a porcentagem de pontos diferentes de zero na matriz documento-palavra.

Como a maioria das células nessa matriz será zero, estou interessado em saber qual porcentagem de células contém valores diferentes de zero.
'''
# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Esparsicidade: ", round(((data_dense > 0).sum()/data_dense.size)*100, 2), "%")

lda_model = LatentDirichletAllocation(n_components=5,              
                                      max_iter=10,               
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=128,            
                                      evaluate_every = -1,       
                                      n_jobs = -1             
                                     )
lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)

# Probabilidade logaritmica: quanto maior melhor
print("probabilidade logaritmica: ", round(lda_model.score(data_vectorized), 2))
# Perplexidade: menor melhor.  exp(-1. * log-Probabilidade logaritmica por palavra)
print("Perplexidade: ", round(lda_model.perplexity(data_vectorized), 2))
print("Parâmetros:")
print(lda_model.get_params())

# Definindo os parâmetros de pesquisa
search_params = {'n_components': [5, 10, 15], 'learning_decay': [.5, .7, .9]}
# Inicializando o modelo
lda = LatentDirichletAllocation()
# Inicializando o método GridSearch
model = GridSearchCV(lda, param_grid=search_params)
# Realizando a pesquisa em grade
model.fit(data_vectorized)

# Melhor modelo
best_lda_model = model.best_estimator_
# Hiperparâmetros do modelo
print("Melhores parâmetros: ", model.best_params_)
# probabilidade logarítmica
print("Melhor score de probabilidade logarítmica: ", model.best_score_)
# Perplexidade
print("Perplexidade do modelo: ", best_lda_model.perplexity(data_vectorized))

import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt 
results = pd.DataFrame(model.cv_results_)

current_palette = sns.color_palette("Set2", 3)

plt.figure(figsize=(12,8))

sns.lineplot(data=results,
             x='param_n_components',
             y='mean_test_score',
             hue='param_learning_decay',
             palette=current_palette,
             marker='o'
            )
plt.show()

# Criando documento - Matriz de tópicos
lda_output = best_lda_model.transform(data_vectorized)
# Nome das colunas
topicnames = ["Tópico " + str(i) for i in range(best_lda_model.n_components)]
# Nome dos índices
docnames = ["Documento: " + str(i) for i in range(len(df['reclamacao_completa']))]
# Criando o DataFrame
df_document_topic = pd.DataFrame(np.round(lda_output, 6), columns=topicnames, index=docnames)
# Selecionando o tópico dominante em cada documento
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic
# Estilo
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)
def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)
# Aplicando estilo
df_document_topics = df_document_topic.style.applymap(color_green).applymap(make_bold)
df_document_topics_first5 = df_document_topic[:5].style.applymap(color_green).applymap(make_bold)
df_document_topics_first5

# Distribuição de documentos em cada tópico
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Número de Documentos")
df_topic_distribution.columns = ['Número do Tópico','Número de Documentos']
print(df_topic_distribution)
df_topic_distribution = (df_document_topic['dominant_topic'].value_counts()/len(df['reclamacao_completa'])).reset_index(name="% de Documentos")
df_topic_distribution.columns = ['Número do Tópico','% de Documentos']
print(df_topic_distribution)

# Plotando gráfico de distribuição
df_topic_distribution_transposto =  df_topic_distribution.T
df_topic_distribution_transposto
df_topic_distribution_transposto.columns = df_topic_distribution_transposto.iloc[0]
df_topic_distribution_transposto = df_topic_distribution_transposto[1:]
df_topic_distribution_transposto
df_topic_distribution_transposto.plot.barh(stacked=True,figsize=(15, 8))

# Top palavras por tópico
vocab = vectorizer.get_feature_names()
# Dados vetorizados
topic_words = {}
n_top_words = 5
for topic, comp in enumerate(best_lda_model.components_):  
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [vocab[i] for i in word_idx]    
for topic, words in topic_words.items():
    print('Topic: %d' % topic)
    print('  %s' % ', '.join(words))

# Selecionando documentos aleatórios para a análise quantitativa
import random
random.seed(123456) # Permite replicabilidade do experimento
# Agrgando ao dataframe original a classificação de tópico resultado do LDA
df['topic_classification'] = dominant_topic
# Todos os tópicos possíveis gerados pelo LDA
possible_topics = np.sort(df['topic_classification'].unique())
# Quantidade de reclamações para análise qualitativa
n_reclamacoes = 15
for topic in possible_topics:
  print("{} reclamações para o tópico {}".format(n_reclamacoes, topic))
  for i in range(n_reclamacoes):
    recl_aleatoria = random.randint(0,len(df[df['topic_classification'] == topic]))
    print('\tID Reclamação {}: {}\n'.format(df[df['topic_classification'] == topic]['ID'].values[recl_aleatoria], df[df['topic_classification'] == topic]['reclamacao_completa'].values[recl_aleatoria]))

# Top 5 localidades por tópico
# Transformando objeto style em um dataframe pandas
df2 = pd.DataFrame(data=df_document_topics.data, columns=df_document_topics.columns)
df2.head()
# Associando as localidades aos tópicos 
df2['Local do reclamante'] = df['Local do reclamante'].tolist()
df2.head()
# Localidade que mais aparece dentro de cada tópico
df2.groupby(["dominant_topic"])['Local do reclamante'].agg(pd.Series.mode).to_frame()
for i in range(best_lda_model.n_components):
  print('Tópico: '+str(i))
  print(df2[df2["dominant_topic"]==i].groupby(['Local do reclamante']).size().sort_values(ascending=False)[:5])

# Top 5 notas por tópico
# Transformando objeto style em um dataframe pandas
df2 = pd.DataFrame(data=df_document_topics.data, columns=df_document_topics.columns)
df2.head()
# Associando as notas aos tópicos 
df2['Nota'] = df['Nota'].tolist()
df2.head()
# Nota que mais aparece dentro de cada tópico
df2.groupby(["dominant_topic"])['Nota'].agg(pd.Series.mode).to_frame()
for i in range(best_lda_model.n_components):
  print('Tópico: '+str(i))
  print(df2[df2["dominant_topic"]==i].groupby(['Nota']).size().sort_values(ascending=False)[:5]) 

# Avaliações por tópico
# Transformando objeto style em um dataframe pandas
df2 = pd.DataFrame(data=df_document_topics.data, columns=df_document_topics.columns)
df2.head()
# Associando as notas aos tópicos 
df2['Voltaria fazer Negocio'] = df['Voltaria fazer Negocio'].tolist()
df2.head()
# Resposta que mais aparece dentro de cada tópico
df2.groupby(["dominant_topic"])['Voltaria fazer Negocio'].agg(pd.Series.mode).to_frame()
for i in range(best_lda_model.n_components):
  print('Tópico: '+str(i))
  print(df2[df2["dominant_topic"]==i].groupby(['Voltaria fazer Negocio']).size().sort_values(ascending=False)[:5])