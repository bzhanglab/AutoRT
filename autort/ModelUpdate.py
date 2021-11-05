import tensorflow as tf
import json
import os
from shutil import copyfile

class ModelUpdate:

    def __init__(self, out_dir:str, max_x_length=0, add_mods=None):
        self.out_dir = out_dir
        self.max_x_length = max_x_length
        self.add_mods = add_mods

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def update_model_input_shape(self, models_file:str):

        print("Change the max length of peptide setting in models:")

        # this will destroy the weights in the models
        with open(models_file, "r") as read_file:
            model_list = json.load(read_file)

        print("The current peptide length: %s" % str(model_list['max_x_length']))

        with open(models_file, "r") as read_file:
            new_model_list = json.load(read_file)

        model_folder = os.path.dirname(models_file)

        for (name, dp_model_file) in model_list['dp_model'].items():
            dp_model_path = model_folder + "/" + os.path.basename(dp_model_file)
            model = tf.keras.models.load_model(dp_model_path)
            config = json.loads(model.to_json())
            input_layer = str(config['config']['layers'][0]['class_name'])
            if "inputlayer" in input_layer.lower():
                config['config']['layers'][0]['config']["batch_input_shape"][1] = self.max_x_length
            else:
                print("Do not find InputLayer!!!")

            embedding_layer = str(config['config']['layers'][1]['class_name'])
            if "embedding" in embedding_layer.lower():
                config['config']['layers'][1]['config']["batch_input_shape"][1] = self.max_x_length
                if "input_length" in config['config']['layers'][1]['config']:
                    config['config']['layers'][1]['config']['input_length'] = self.max_x_length
            else:
                print("Do not find Embedding!!!")

            new_model = tf.keras.models.model_from_json(json.dumps(config))
            new_model.compile(optimizer="adam", loss="mean_squared_error")
            new_model_file = self.out_dir + "/" + os.path.basename(dp_model_file)
            new_model.save(new_model_file)

        new_model_list['max_x_length'] = self.max_x_length
        print("The peptide length for new models: %s" % str(new_model_list['max_x_length']))
        # save result
        model_json = self.out_dir + "/model.json"
        with open(model_json, 'w') as f:
            json.dump(new_model_list, f, indent=2)

        aa_file = os.path.basename(model_list['aa'])
        aa_file = model_folder + "/" + aa_file
        new_aa_file = self.out_dir + "/" + os.path.basename(model_list['aa'])
        copyfile(aa_file, new_aa_file)

        return model_json
