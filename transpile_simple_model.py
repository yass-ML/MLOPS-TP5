import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class LinearModelTranspiler:
    def __init__(self, model_path:str, output_c_file:str = "transpiled_model.c"):
        if model_path is None or model_path == "":
            raise ValueError("No model path provided")
        if output_c_file is None or output_c_file == "":
            raise ValueError("No output path for transpilation provided")
        self.model_path = model_path
        self.output_c_file = output_c_file
        self.reg_type = None
        self.params = None
        self.c_code = None

    def _extract_linear_model_params(self, model):
        if isinstance(model,LinearRegression):
            self.reg_type = "linear"
            self.params = {"intercept": model.intercept_, "coefficients": model.coef_, "n_features": len(model.coef_)}
        elif isinstance(model,LogisticRegression) and len(model.intercept_) == 1: # Must be a binary Logistic Regression
            self.reg_type = "logistic"
            self.params = {
            "intercept": model.intercept_[0],
            "coefficients": model.coef_[0],
            "classes": model.classes_,
            "n_features": len(model.coef_[0])
            }
        elif isinstance(model, DecisionTreeClassifier):
            self.reg_type = "decision_tree_classifier"
            self.params = {
                "tree": model.tree_,
                "n_features": model.n_features_in_,
                "n_classes": model.n_classes_,
                "classes": model.classes_
            }
        elif isinstance(model, DecisionTreeRegressor):
            self.reg_type = "decision_tree_regressor"
            self.params = {
                "tree": model.tree_,
                "n_features": model.n_features_in_
            }
        else:
            raise ValueError(f"Unknown model type: {type(model).__name__}. This transpiler recognizes linear regression, binary logistic regression, or decision trees")


    def _generate_tree_c_code(self, tree, node_id=0, depth=0):
            """
            Recursively generate C code for decision tree nodes.
            
            Args:
                tree: sklearn tree_ object
                node_id: current node index
                depth: current depth (for indentation)
            
            Returns:
                C code string for this node and its children
            """
            indent = "    " * depth
            
            # Check if leaf node
            if tree.feature[node_id] == -2:  # -2 indicates leaf node
                # For classification, return the majority class
                if self.reg_type == "decision_tree_classifier":
                    # Get the class with highest count
                    class_counts = tree.value[node_id][0]
                    predicted_class = int(np.argmax(class_counts))
                    return f"{indent}return {predicted_class}.0f;\n"
                else:
                    # For regression, return the mean value
                    value = tree.value[node_id][0][0]
                    return f"{indent}return {value}f;\n"
            
            # Internal node - generate if-else condition
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            code = f"{indent}if (features[{feature_idx}] <= {threshold}f) {{\n"
            code += self._generate_tree_c_code(tree, left_child, depth + 1)
            code += f"{indent}}} else {{\n"
            code += self._generate_tree_c_code(tree, right_child, depth + 1)
            code += f"{indent}}}\n"
            
            return code

    def load_model(self):
        print(f"Loading model from {self.model_path}...")
        model = joblib.load(self.model_path)
        self._extract_linear_model_params(model)


    def save_transpiled_model(self):
        if self.reg_type is None or self.params is None:
            raise RuntimeError("No model has been loaded for transpilation")
        if self.c_code is None:
            raise RuntimeError("No transpiled C code found")
        if self.output_c_file is None:
            raise RuntimeError("No path to save the transpiled model")
        with open(self.output_c_file, 'w') as f:
            f.write(self.c_code)
    
        
        print(f"\nC code generated and saved to: {self.output_c_file}")


    def transpile(self):
        self.load_model()
        self.generate_c_code()
        self.save_transpiled_model()

    def generate_c_code(self):
        if self.reg_type is None or self.params is None:
            raise RuntimeError("No model has been loaded for transpilation")
        

        # Handle decision trees
        if self.reg_type in ["decision_tree_classifier", "decision_tree_regressor"]:
            tree = self.params["tree"]
            n_features = self.params["n_features"]
            
            model_desc = f"// Generated {self.reg_type} model\n"
            model_desc += f"/* n_features: {n_features} */\n"
            if self.reg_type == "decision_tree_classifier":
                model_desc += f"/* n_classes: {self.params['n_classes']} */\n"
                model_desc += f"/* classes: {self.params['classes']} */\n"
            
            tree_code = self._generate_tree_c_code(tree)
            
            self.c_code = f"""#include <stdio.h>
#include <stdlib.h>

{model_desc}

float prediction(float *features) {{
{tree_code}}}
"""
            return
        
        # Handle linear models
        coef_str = ", ".join([f"{coef}f" for coef in self.params["coefficients"]])
        intercept = self.params["intercept"]
        n_features = self.params["n_features"]

    #     sigmoid_str ="""
    # float exp_approx(float x, int n_term){
    #     if (n_term < 0)
    #         return 0.0f;
    #     float exp_x = 1.0f;
    #     float fact_acc = 1.0f;
    #     float x_i_acc = 1.0f;
    #     for (int i = 1; i <= n_term; i++){
    #         fact_acc *= i;
    #         x_i_acc *= x;
    #         exp_x += x_i_acc / fact_acc;
    #     }
    #     return exp_x;
    # }
        


    # float sigmoid(float x){
    #     return 1.0f / (1.0f + exp_approx(-x, 48));
    # }
    # """
        sigmoid_str ="""#include <math.h>
        float sigmoid(float x){
        return 1.0f / (1.0f + expf(-x));
        }
        """
        model_desc_str = f"// Generated {self.reg_type} regression model\n" + "\n".join([f"/* {key}: {value} */" for key,value in self.params.items()])
        
        
        self.c_code = f"""#include <stdio.h>
#include <stdlib.h>

{sigmoid_str}

{model_desc_str}

float prediction(float *features) {{
    // Intercept (bias term)
    float result = {intercept}f;
    
    // Coefficients
    float coefficients[] = {{{coef_str}}};
    int n_features = {n_features};

    int logistic = {int(self.reg_type == "logistic")};
    
    // Calculate: result = intercept + sum(coef_i * feature_i)
    for (int i = 0; i < n_features; i++) {{
        result += coefficients[i] * features[i];
    }}
    if (logistic == 0)
        return result;
    else{{
        float classif_res = sigmoid(result);
        if (classif_res <= 0.5f)
            return 0.0f;
        else
            return 1.0f;
    }}
}}
"""

