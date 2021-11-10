# Importação das bibliotecas
from selenium import webdriver
import selenium
import time
from unicodedata import normalize
import pandas
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime
def Scrap_etapa_1 (url, arquiv_saida):
    # Inicializando o Web driver Selenium
    driver = webdriver.Chrome()    
    # Informa o número de páginas a serem visitadas
    paginas = 101
    # Informa o url da primeira página a ser visitada
    driver.get(url)
    # Criando as colunas que receberão os dados
    titulo = []
    url_reclamacoes =[]
    # Criação do looping 
    for contador_paginas in range(0,paginas):
        time.sleep(5)
        # Obtem o título das reclamações
        sec_titulo = driver.find_elements_by_css_selector('p.text-title')
        print(sec_titulo)
        # Formata o título
        for i in sec_titulo:
            texto = normalize('NFKD', i.text).encode('ASCII', 'ignore').decode('ASCII')
            print(texto)
            titulo.append(texto)
        # Fim do título
        # Link das reclamações
        url_reclamacao = driver.find_elements_by_css_selector('a.link-complain-id-complains')
        for i in url_reclamacao:
            url_reclamacoes.append(i.get_attribute('href'))
        # Fim do link das reclamações
        # Simulação clique para a próxima página
        try:
            action = webdriver.ActionChains(driver)
            next_button = driver.find_element_by_xpath('//li[@class="pagination-next ng-scope"]/a')
            action.move_to_element(next_button)
            action.perform()
            time.sleep(2)
            action.move_to_element(next_button)
            action.click().perform()
        except NoSuchElementException:
            print("Não há mais páginas")
            break
    print("Exportação")
    # Titulo e link
    df = pandas.DataFrame(data={"Titulo": titulo, "link_reclamacao_completa":url_reclamacoes})
    # Exportação
    suffix_date = datetime.today().strftime('%Y%m%d_%H%M')
    file = arquiv_saida + "_" + suffix_date + ".csv"
    df.to_csv(file, sep='~',index=False, encoding='utf-8')
    driver.quit
    print("Acabou o processamento - Arquivo de saida gerado: {}".format(file))
# Listando empresas
empresas = {
    'ubereats': 'https://www.reclameaqui.com.br/empresa/uber-eats/lista-reclamacoes/',
    'rappi': 'https://www.reclameaqui.com.br/empresa/rappi/lista-reclamacoes/',
    'ifood':  'https://www.reclameaqui.com.br/empresa/ifood/lista-reclamacoes/'
}
# Executando o Algoritmo
for empresa in empresas:
    Scrap_etapa_1(empresas[empresa], empresa)
