import pandas as pd
import time
from selenium import webdriver
import selenium
from unicodedata import normalize
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime

def Scrap_etapa_3 (driver, empresa, arquiv_entrada):
    df = pd.read_csv(arquiv_entrada, usecols=['link_reclamacao_completa'], sep='~')
# Criação das listas que receberão os dados das reclamações
    url_reclamacao = []
    titulo = []
    localizacao = []
    id_reclamacao = []
    data_hora = []
    problema = []
    produto = []
    categoria =[]
    status = []
    reclamacao = []
    consideracao_final = []
    voltaria_fazer_negocio = []
    nota = []

    contador = 1

    for url in df['link_reclamacao_completa']:
        try:
            driver.get(url)
        except:
            print('URL: '+ url +' esta com problemas')
        
        print("Começando: " + url)
        time.sleep(2)

        sec_url = url
        sec_id = None
        sec_titulo = None
        sec_localizacao = None
        sec_data = None
        sec_categoria = None
        sec_produto = None
        sec_problema = None
        sec_reclamacao = None
        sec_status = None
        sec_voltaria_fazer_negocio = None
        sec_consideracao_final = None
        sec_nota = None

        try:
            sec_id = driver.find_element_by_xpath("//span[@data-testid='complaint-id']")
        except NoSuchElementException:
            print("sec_id vazia")
        try:
            sec_titulo = driver.find_element_by_xpath("//h1[@data-testid='complaint-title']")
        except NoSuchElementException:
            print("sec_titulo vazia")
        try:    
            sec_localizacao = driver.find_element_by_xpath("//span[@data-testid='complaint-location']")
        except NoSuchElementException:
            print("sec_localizacao vazia")
        try:
            sec_data = driver.find_element_by_xpath("//span[@data-testid='complaint-creation-date']")
        except NoSuchElementException:
            print("sec_data vazia")
        try:
            sec_categoria = driver.find_element_by_xpath("//li[@data-testid='listitem-categoria']/div/a")
        except NoSuchElementException:
            print("sec_categoria vazia")
        try:
            sec_produto = driver.find_element_by_xpath("//li[@data-testid='listitem-produto']/div/a")
        except NoSuchElementException:
            print("sec_produto vazia")
        try:
            sec_problema = driver.find_element_by_xpath("//li[@data-testid='listitem-problema']/div/a")
        except NoSuchElementException:
            print("sec_problema vazia")
        try:
            sec_reclamacao = driver.find_element_by_xpath("//p[@data-testid='complaint-description']")
        except NoSuchElementException:
            print("sec_reclamacao vazia")
        try:
            sec_status = driver.find_element_by_xpath("//div[@data-testid='complaint-status']/img")
        except NoSuchElementException:
            print("sec_status vazia")
        try:
            sec_voltaria_fazer_negocio = driver.find_element_by_xpath("//div[@data-testid='complaint-deal-again']")
        except NoSuchElementException:
            print("sec_voltaria_fazer_negocio vazia")
        try:
            sec_consideracao_final = driver.find_element_by_xpath("//div[@data-testid='complaint-interaction']/div[@type='FINAL_ANSWER']/following-sibling::p")
        except NoSuchElementException:
            print("sec_consideracao_final vazia")
        try:
            sec_nota = driver.find_element_by_xpath("//span[contains(text(), 'Nota do atendimento')]/following-sibling::div/div")
        except NoSuchElementException:
            print("sec_nota vazia")

        print("Registros :"+str(contador),end=" ")
        s_url = sec_url
        s_id_reclamacao = "-1" #   id igual a menos -1 indica erro           
        if sec_id is not None:
            s_id_reclamacao = sec_id.text.replace('ID: ', '')
            print(s_id_reclamacao +" - ", end=" ")
        s_titulo = url
        if sec_titulo is not None:
           s_titulo = normalize('NFKD', sec_titulo.text).encode('ASCII', 'ignore').decode('ASCII')
           print(s_titulo)
        s_localizacao = "SL"
        if sec_localizacao is not None:
            s_localizacao = sec_localizacao.text
        s_data_hora = "000000"
        if sec_data is not None:
            s_data_hora = sec_data.text
        s_produto = 'NC'
        if sec_produto is not None:
            s_produto = sec_produto.text
        s_problema = 'NC'
        if sec_problema is not None:
            s_problema = sec_problema.text
        s_categoria = 'NC'
        if sec_categoria is not None:
            s_categoria = sec_categoria.text
            s_status = "NC"
        if sec_status is not None:
            s_status = sec_status.get_property('alt')
        s_reclamacao = "Sem reclamacao"
        if sec_reclamacao is not None:
            s_reclamacao = sec_reclamacao.text
        s_consideracao_final = 'Sem considerações finais'
        if sec_consideracao_final is not None:
           s_consideracao_final = sec_consideracao_final.text
        s_voltaria_fazer_negocio = 'Não avaliado'
        if sec_voltaria_fazer_negocio is not None:
            s_voltaria_fazer_negocio = sec_voltaria_fazer_negocio.text
        s_nota = 'Não avaliado'
        if sec_nota is not None:
            s_nota  = sec_nota.text

        #Atribuição dos dados as variáveis
        url_reclamacao.append(s_url) 
        id_reclamacao.append(s_id_reclamacao)
        titulo.append(s_titulo)
        localizacao.append(s_localizacao)
        data_hora.append(s_data_hora)
        problema.append(s_problema)
        produto.append(s_produto) 
        categoria.append(s_categoria)
        status.append(s_status)
        reclamacao.append(s_reclamacao)
        consideracao_final.append(s_consideracao_final)
        voltaria_fazer_negocio.append(s_voltaria_fazer_negocio)
        nota.append(s_nota)
        contador +=1
    print("Exportação")

    df = pd.DataFrame(data={"URL":url_reclamacao,"ID":id_reclamacao, "Titulo": titulo, "Data Reclamação": data_hora, "Local do reclamante":localizacao, "reclamacao_completa":reclamacao, "Status": status,
                            "Consideracao_Final":consideracao_final,"Problema":problema, "Produto":produto, "Categoria":categoria, "Nota": nota, "Voltaria fazer Negocio":voltaria_fazer_negocio})

    #Arquivo de saída 
    suffix_date = datetime.today().strftime('%Y%m%d_%H%M')
    file = "./"+empresa+"_dados_" + suffix_date + ".csv"
    df.to_csv(file, sep=';',index=False, encoding='utf-8')

#abre o navegador
driver = webdriver.Chrome()

empresas = {
    'UberEats': 'ConsolidadoUber03-10-21.csv',
    'Rappi': 'ConsolidadoRappi03-10-21.csv',
    'iFood':  'ConsolidadoIfood03-10-21.csv'
}

for empresa in empresas:
    print(empresa, empresas[empresa])
    Scrap_etapa_3(driver, empresa, empresas[empresa])

# fecha o navegador
driver.quit()