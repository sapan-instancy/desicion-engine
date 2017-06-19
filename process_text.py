import pandas as pd
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup

pd.set_option('display.max_columns', None)

# id pageid poid value topic
def process_text(directory_path):
    pageDf, pageCompDf, contentLibDf = xml_to_dataframe(directory_path)
    #pageDf: (['name', 'pageId', 'url'],
    #pageCompDf: (['pageId', 'poid', 'type'],
    #contentLibDf: (['id', 'value'], dtype='object')
    normDataDf = pd.merge(pageCompDf, contentLibDf, how='inner', left_on='poid', right_on='id')
    normDataDf.drop('id', axis=1, inplace=True)
    return normDataDf


def xml_to_dataframe(dir_path):

    tree = ET.parse(os.path.join(dir_path, 'content.xml'))
    root = tree.getroot()
    pageList = []
    for page in root.iter('page'):
        pageList.append({ 'pageId': page.attrib['id'], 'name' : page.attrib['name'], 'url' : page.attrib['url'] })
    pageDf = pd.DataFrame(data=pageList)

    pageCompList = []
    for record in pageList:
        tree = ET.parse(os.path.join(dir_path, record['url']))
        root = tree.getroot()

        for po in root.iter('pageobject'):
            if po.attrib['type'] in ['text', 'image']:
                pageCompList.append({ 'pageId' : record['pageId'], 'poid': po.attrib['poid'], 'type': po.attrib['type'] })
    pageCompDf = pd.DataFrame(data=pageCompList)

    contentLibList =[]
    tree = ET.parse(os.path.join(dir_path, 'Contentlibrary.xml'))
    root = tree.getroot()

    for po in root.iter('pageobject'):
        if po.attrib['type'] in ['text', 'image']:
            if po.attrib['type'] in ['text']:
                contentLibList.append({ 'id': po.attrib['id'], 'value': extract_text_from_pageobject(po.text) })
            else:
                contentLibList.append({'id': po.attrib['id'], 'value': po.text })
    contentLibDf = pd.DataFrame(data=contentLibList)

    return pageDf, pageCompDf, contentLibDf


def extract_text_from_pageobject(value):
    x = value
    if x != None:
        if x.find('<') != -1:
            soup = BeautifulSoup(x, 'html.parser')
            x = soup.get_text()
    return x



# directory_path = r'D:\MyDev\Working\8.4\Content\Sample Content\content\pages'
# print(process_text(directory_path)['value'])