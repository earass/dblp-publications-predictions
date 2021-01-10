from lxml import etree
import csv
import re


def extract_feature(elem, features):
    attribs = {}
    for feature in features:
        attribs[feature] = []
    for sub in elem:
        if sub.tag not in features:
            continue
        if sub.tag == 'title':
            text = re.sub("<.*?>", "", etree.tostring(sub).decode('utf-8')) if sub.text is None else sub.text
        else:
            text = sub.text
        if text is not None and len(text) > 0:
            attribs[sub.tag] = attribs.get(sub.tag) + [text]
    return attribs


def parse_entity(input_path, save_path, type_name, features):
    all_elements = ["article", "inproceedings", "proceedings", "book", "incollection", "phdthesis", "mastersthesis",
                    "www"]
    results = []
    tree = etree.iterparse(source=input_path, dtd_validation=True, load_dtd=True)
    for _, elem in tree:
        if elem.tag in type_name:
            attrib_values = extract_feature(elem, features)
            results.append(attrib_values)
        elif elem.tag not in all_elements:
            continue
        # clearing element
        elem = None
    with open(save_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(features)
        for record in results:
            row = ['|'.join(v) for v in list(record.values())]
            writer.writerow(row)


def parse_xml():
    all_features = ["address", "author", "booktitle", "cdrom", "chapter", "cite", "crossref", "editor", "ee", "isbn",
                    "journal", "month", "note", "number", "pages", "publisher", "school", "series", "title", "url",
                    "volume", "year"]
    input_path = 'data/dblp.xml'
    save_path = 'data/articles_2.csv'
    type_name = ['article']
    features = all_features
    parse_entity(input_path, save_path, type_name, features)


if __name__ == '__main__':
    parse_xml()
