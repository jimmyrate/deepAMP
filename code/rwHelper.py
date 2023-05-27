import os
import csv
import pickle

class csvHelper:
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def writeRows(self, data, mode='a'):
        filedNames = self.fileCheck(data, mode)
        if isinstance(data, dict):
            data = [data]
        with open(self.file_name,'a') as f:
            f_csv = csv.DictWriter(f, filedNames)
            f_csv.writerows(data)
    
    def fileCheck(self, data, mode):
        data = data if isinstance(data, dict) else data[0]
        header = data.keys()
        if not os.path.exists(self.file_name) or mode=='w':
            with open(self.file_name,'w') as f:
                f_csv = csv.DictWriter(f, header)
                f_csv.writeheader()
        return header

    def readDict(self)->[{},{},...]:
        if not os.path.exists(self.file_name):
            return
        with open(self.file_name,'r') as f:
            rows = list(csv.reader(f))
            header = rows[0]
            contents = [{k: row[idx] for idx, k in enumerate(header)} for row in rows[1:]]
            # contents = list(csv.DictReader(f))
            return contents

class dicTxtHelper:
    def __init__(self, file_name):
        '''
        file_name is a "xxx.txt" with a content like
        content:
            attr1:xxx
            attr2:xxx
        '''
        super().__init__()
        self.file_name = file_name
    
    def writeDict(self, data:dict):
        with open(self.file_name, 'w') as f:
            text = [':'.join([str(k),str(v)]) for k, v in data.items()]
            text = '\n'.join(text)
            f.write(text)
    
    def readDict(self):
        data = dict()
        try:
            with open(self.file_name, 'r') as f:
                text_list = f.read().strip().split('\n')
                for line in text_list:
                    line = line.split(':')
                    data[line[0]] = line[1]
        except:
            return None
        return data
            
class objHelper:
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def save_obj(self, obj):
        """save obj with pickle"""
        with open(self.file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self):
        """load a pickle object"""
        try:
            with open(self.file_name, 'rb') as f:
                return pickle.load(f)
        except:
            return None

class lineTxtHelper:
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name
    
    def addOneLine(self, data):
        with open(self.file_name,'a') as f:
            text = str(data)+'\n'
            f.write(text)
    
    def writeLines(self, data_list):
        with open(self.file_name,'w') as f:
            text = '\n'.join(str(i) for i in data_list)
            f.write(text)

    def readLines(self):
        try:
            with open(self.file_name,'r') as f:
                data_list = f.read().strip().split('\n')
                return data_list
        except:
            return None
        return None


