import json


class Ontology:
    def __init__(self, filename):
        self.ontology_file_name = filename
        self.load_ontology()

    def load_ontology(self):
        with open(self.ontology_file_name) as ont_file:
            self.ontology = json.load(ont_file)
            self.ontology['system_requestable'] = list(self.ontology['informable'].keys())
