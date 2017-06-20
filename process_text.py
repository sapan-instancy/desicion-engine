import pandas as pd
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup

pd.set_option('display.max_columns', None)

# this file processes xml's present in Content folder of a course to a dataframe in the form:
# ['pageId', 'elementid', 'type', 'value'] value contains text
def process_text(content_directory_path, output_file_fullname=None):
    pageDf, pageCompDf, contentLibDf = xml_to_dataframe(content_directory_path)
    #dataframe pageDf columns: (['name', 'pageId', 'url'],
    #dataframe pageCompDf columns: (['pageId', 'poid', 'elementid', 'type'],
    #dataframe contentLibDf columns: (['id', 'value'], dtype='object')
    normDataDf = pd.merge(pageCompDf, contentLibDf, how='inner', left_on='poid', right_on='id')
    normDataDf.drop('id', axis=1, inplace=True)
    normDataDf.drop('poid', axis=1, inplace=True)
    if output_file_fullname != None:
        normDataDf.to_csv(output_file_fullname)
    return normDataDf

#this is a private method that parses xml's and creates the required dataframes
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
                pageCompList.append({ 'pageId' : record['pageId'], 'poid': po.attrib['poid'], 'elementid': po.attrib['id'], 'type': po.attrib['type'] })
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


#this private method parses the node object containing text and extracts useful text without html markup out of it.
def extract_text_from_pageobject(value):
    x = value
    if x != None:
        if x.find('<') != -1:
            soup = BeautifulSoup(x, 'html.parser')
            x = soup.get_text()
    return x


## to test this file, uncomment the lines below
# directory_path = r'D:\MyDev\Working\8.4\Content\Sample Content\content\pages'
# output_file_fullname=r'C:\Users\testuser\Documents\desicion_engine_1\tmp\processed_text_ergo.csv'
# print(process_text(directory_path)['value'])