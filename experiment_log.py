class PytorchExperimentLogger(object):
    def __init__(self, saveDir, fileName,ShowTerminal=False):

        self.saveFile = saveDir + r"/" + fileName +".txt"
        self.ShowTerminal = ShowTerminal

    def print(self, strT):
        #
        if self.ShowTerminal:
            print(strT)
        f = open(self.saveFile, 'a')
        f.writelines(strT+'\n')
        f.close()
