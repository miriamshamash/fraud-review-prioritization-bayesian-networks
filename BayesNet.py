import json

class BayesNet:
    def __init__(self, json_file_path=None):
        if json_file_path:
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            self.data = data
            self.tables = data['tables']
            self.nodes = data['nodes']
            self.parents = data['parents']

    def query_prob(self, variable, var_value, evidence):
        """
        Gets the probability a variable taking on var_value given specified evidence.

        Args:
            variable (String) : The variable we're querying
            var_value (String) : The value that variable is taking on
            evidence (Dict): Specified evidence for that variable (or empty if variable has no parents).
                            Keys are names of variables, and values are a specific outcome value.

        Returns:
            float : The probabilty of a variable taking on var_value given specified evidence,
                    as given directly by the conditional probability tables.
                    If insufficent evidence is given, return None.

        """

        # check that evidence contains all parent values, 
        for parent in self.parents[variable]:
            if parent not in evidence:
                return None

        # build ordered list of parent values
        parent_values = []
        for parent in self.parents[variable]:
            parent_values.append(evidence[parent])

        # find the table row that matches these parent values
        for row in self.tables[variable]:
            parent_assignment = row[0]
            probabilities = row[1]

            if parent_assignment == parent_values:
                possible_values = self.nodes[variable]

                index = 0
                for val in possible_values:
                    if val == var_value:
                        return probabilities[index]
                    index += 1

                return None

        return None

        
    def enumerate_all(self, vars, evidence, index=0):
        """
        Recursivley caculates the joint probability of variables taking on the values specified in evidence.

        Args:
            variable (List[String]) : The variables present in the bayes net, specified in topological order.
            evidence (Dict): Dictionary representing specific outcomes.
                            Keys are names of variables, and values are a specific outcome value.

        Returns:
            float : The result of the joint probability query for P(evidence).

        """
        # Base case        
        if index == len(vars):
            return 1.0

        current_var = vars[index]

        # Current variable in evidence -> multiplication
        if current_var in evidence:

            parent_evidence = {}
            for parent in self.parents[current_var]:
                parent_evidence[parent] = evidence[parent]

            probability = self.query_prob(current_var, evidence[current_var], parent_evidence)
            return probability * self.enumerate_all(vars, evidence, index + 1)

        # Current variable not in evidence -> addition
        else:
            total = 0.0

            # Try each possible assignment of current_var
            for possible_value in self.nodes[current_var]:

                new_evidence = evidence.copy()
                new_evidence[current_var] = possible_value

                parent_evidence = {}
                for parent in self.parents[current_var]:
                    parent_evidence[parent] = new_evidence[parent]

                probability = self.query_prob(current_var, possible_value, parent_evidence)
                total += probability * self.enumerate_all(vars, new_evidence, index + 1)

            return total

    def enumerate_ask(self, query, evidence):
        """
        Calculates the distribution of P(query | evidence) for every possible value query can take on.

        Args:
            query (String) : The variable we wish to know the distribution of.
            evidence (Dict): The evidence specified.
                            Keys are names of variables, and values are a specific outcome value.

        Returns:
            Dictionary : A dictionary representing the entire distribution of P(query | evidence).
                        Keys are the possible outcomes of query, values are the probabilities of each outcome.

        """

        distribution = {}
        nodes = list(self.nodes.keys())

        # Compute unnormalized probabilities using enumerate_all
        for value in self.nodes[query]:
            new_evidence = evidence.copy()
            new_evidence[query] = value

            prob = self.enumerate_all(nodes, new_evidence, index=0)
            distribution[value] = prob

        # Normalize dict
        total = sum(distribution.values())
        for value in distribution:
            distribution[value] = distribution[value] / total

        return distribution


def main():
    bn = BayesNet("./nets/sprinkler.json")

    bn.query_prob("Sprinkler", "T", {"Cloudy": "F"})

    # Feel free to write down here for whatever input and testing you wish.
    # Make sure to check "./tests" for outputs

if __name__ == "__main__":
    main()  