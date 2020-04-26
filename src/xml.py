import os
import pandas as pd
import re
import xml.etree.ElementTree as ET


def read_xml(file):
    ET.fromstring(file)

def real_xml(path = 'iniciativas08/'):
    files = []
    df = []
    for _,_,f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(path + file)
    files = pd.DataFrame(files, columns = ['file'])#.head(5)
    #print(files)

    files['intervenciones'] = files['file'].apply(
        lambda f: ET.parse(f).findall('iniciativa/intervencion')
    )
    
    for _,intervenciones in files['intervenciones'].iteritems():
        for intervencion in intervenciones:
            df.append({
                'name': intervencion.findtext('interviniente'),
                'text': '. '.join([str(parrafo.text) for parrafo in intervencion.findall('discurso/parrafo')])
            })
    df = pd.DataFrame(df, columns = ['name','text'])

    filters = [line.replace('\n', '').strip() for line in open('replace.txt', 'r', encoding = 'utf8').readlines()]

    my_filter = re.compile('|'.join(map(re.escape, filters)))

    df['name'] = df['name'].apply(lambda name: 
        name.replace(
            "PÉREZ GARCÍA DE PRADO", "PÉREZ GARCÍA DEL PRADO"
        ).replace(
            ',', ''
        ).replace(
            '.', ''
        ).replace(
            '/', ''
        ).strip()
    )

    

    df['name'] = df['name'].apply(lambda name:
        my_filter.sub("", name).strip()
    )

    df.drop(df[df['name'] == ''].index,inplace = True)

    df['text'] = df['text'].apply(lambda text: text.replace('-',''))

    print(df['name'].unique())
    df['name'] = df['name'].apply(lambda name: name.upper())
    df['name'] = df['name'].apply(lambda name: ' '.join(name.split(' ')[:4]))

    print(df['name'].unique().shape)
    print(df['name'].value_counts())
    print(df.shape)

    return(df)

if __name__ == "__main__":
    main()