from enum import Enum 
from collections import namedtuple
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import pickle 
import re 
import networkx as nx

QueryTuple = namedtuple("QueryTuple", ["node", "outage", "subgraph", "ON_OFF"])
Dataset_tensors = namedtuple("Dataset_tensors", ["input", "output", "mask"])

class Outage(Enum) : 
    DV = 0
    TA = 1
    Isolated = 2

def str_to_Outage(string) : 
    assert string in "DVTA#" , f"{string} must be one the 3 outage mode : DV, TA, #"
    if (string == "DV") : 
        return Outage.DV
    elif(string == "TA") : 
        return Outage.TA
    elif(string == "#") : 
        return Outage.Isolated 
    else : 
        raise Exception("Error couldn't find a outage mode")

        
class SubtitleFormatError(Exception) : 
    def __init__(self, string) : 
        super(SubtitleFormatError, self).__init__(string)
        
class NotInSubgraphError(Exception) : 
    def __init__(self, string) : 
        super(NotInSubgraphError, self).__init__(string)
        
class NodeNotFoundError(Exception) : 
    def __init__(self, string, node) : 
        super(NodeNotFoundError, self).__init__(string)
        self.node = node 
        
class Dataset : 
    
    def __init__(self) : 
        self.cl_df = pd.read_csv("cl_standard_content.csv")
        self.cl_info_df = pd.read_csv("cl_standard_index.csv")
        self._topo = pickle.load(open("nb.pickle", "rb"))
        self._getActionDict()
        self._number_of_node_type = 34
        self.target_size = 50
        self.max_in_size = 150
        self.max_out_size = 70
        self.input_size = ((20 + self._number_of_node_type) * 2) + 6
    
    
    def GetNodeType(self, node): 

        ix=0
        match = re.match(r".*\s*DV\s*.*", node, re.I)
        if match : 
            return ix
            
        ix+=1     
        match = re.match(r".*\s*SL\s*.*", node, re.I)
        if match : 
            return ix
        
        ix+=1 
        match = re.match(r".*\s+\b(L[IK]|CK|TI)\b", node, re.I)
        if match : 
            match = re.match(r"DI \d{1,3} (\w{1,3}|F \d{1,2}) (.{7,10}) (L[IK]|CK|TI)", node, re.I)
            if match : 
                line_number = match.group(1) 
                match = re.match(r"([\w\+]{3,5})\s?([\w\+]{3,5})", match.group(2), re.I) 
                if match : 
                    return ix
                    
        ix+=1
        match = re.match(r".*\s*SR\s*.*", node, re.I)
        if match : 
            return ix
        
        ix+=1 
        match = re.match(r".*\s+TFO", node, re.I)
        if match :
            return ix
        
        ix+=1    
        match = re.match(r".* (F?[BR][A-Z]*\d*[A-Z]*|T\d{1,3}) .*BS", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".*\s*SI\s*.*", node, re.I)
        if match : 
            return ix
            
        ix+=1     
        match = re.match(r".*\s*(F [^B].*|AUX) BS", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".*SER DEV", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".* CK", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".* \b\w{10}CK\b", node, re.I)
        if match : 
            return ix
        ix+=1                        
        match = re.match(r".* SCL [A-Z]?\d*[A-Z]?\s*SW", node, re.I)
        if match : 
            return ix
        ix+=1                 
        match = re.match(r"DI .* TI", node, re.I)
        if match : 
            return ix 
        ix+=1     
        match = re.match(r".* \b\w{10}TI\b", node, re.I)
        if match : 
            return ix
        ix+=1                        
        match = re.match(r".* IND SHUNT", node, re.I)
        if match : 
            return ix 
        ix+=1     
        match = re.match(r".* COND \d*\s?[A-Z]?\s?COND", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".* BYPASS SW", node, re.I)
        if match : 
            return ix
        ix+=1    
        match = re.match(r".* S\s*\d(/\w{2,3})? SW", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".* \+T\s?\d+[A-Z]? (\+T\s?\d+[A-Z]? )?BS", node, re.I)
        if match : 
            return ix
        ix+=1               
        match = re.match(r".* \+A\s?0+[A-Z]? (\+A\s?0+[A-Z]? )?BS", node, re.I)
        if match : 
            return ix
        ix+=1              
        match = re.match(r".* \+C\s?\d+[A-Z]? (\+C\s?\d+[A-Z]? )?BS", node, re.I)
        if match : 
            return ix
        ix+=1            
        match = re.match(r".*CLAM \d?\s*SW", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".* SELF \d*\s*SELF", node, re.I)
        if match : 
            return ix
        ix+=1 
        match = re.match(r".* \+T\s?\d+[A-Z]? (\+T\s?\d+[A-Z]? )?BS", node, re.I)
        if match : 
            return ix
        ix+=1     
        match = re.match(r".* T\s?[A-Z]?\d*[A-Z]? .* SELF", node, re.I)
        if match : 
            #occurence_sorted[f"T SELF"] = occurence_sorted.get(f"T SELF", 0) + 1
            return ix 
        ix+=1             
        match = re.match(r".* \b\w{2,10}\s?SELF\b", node, re.I)
        if match : 
            #occurence_sorted[f"XXXXX SELF"] = occurence_sorted.get(f"XXXXX SELF", 0) + 1
            return ix 
        ix+=1     
        match = re.match(r".* F \b\w{4,5}\b BS", node, re.I)
        if match : 
            #occurence_sorted[f"xxxx BS"] = occurence_sorted.get(f"xxxx BS", 0) + 1
            return ix
        ix+=1             
        match = re.match(r".* F ZT [A-Z] BS", node, re.I)
        if match : 
            #occurence_sorted[f"ZT BS"] = occurence_sorted.get(f"ZT BS", 0) + 1
            return ix
        ix+=1     
        match = re.match(r".* IND SERIES", node, re.I)
        if match : 
            #occurence_sorted[f"IND SERIES"] = occurence_sorted.get(f"IND SERIES", 0) + 1
            return ix
        ix+=1         
        match = re.match(r".* CAP SERIES", node, re.I)
        if match : 
            #occurence_sorted[f"CAP SERIES"] = occurence_sorted.get(f"CAP SERIES", 0) + 1
            return ix
        ix+=1     
        match = re.match(r".* CCAT SW", node, re.I)
        if match : 
            #occurence_sorted[f"CCAT"] = occurence_sorted.get(f"CCAT", 0) + 1
            return ix   
        ix+=1 
        match = re.match(r".* FILTRE COND", node, re.I)
        if match : 
            #occurence_sorted[f"FILTRE COND"] = occurence_sorted.get(f"FILTRE COND", 0) + 1
            return ix         
        ix+=1 
        match = re.match(r"MN-\d{2,3}", node, re.I) 
        if match : 
            #occurence_sorted["MN"] = occurence_sorted.get("MN", 0) + 1
            ix+=1 
            return ix
        #Other
        return ix + 1
        
        #raise Exception(f"couldn't get the type of {node}")      
        
        
    #checked        
    def _getActionDict(self): 
        action_list_unique = self.cl_df["text_fr_action"].unique()
        self._action_dict = dict(zip(action_list_unique, range(len(action_list_unique))))

    
    #checked    
    def Number2Vector(self, number, max_length) : 
        lst = list() 
        indx = 0
        digit = number % 10
        vect = np.zeros(0) 
        while  max_length > indx : 
            vect = np.concatenate([tf.keras.utils.to_categorical(digit, 10), vect]) 

            indx+= 1
            digit = (number//(10**indx))% 10 

        return vect 

    def Checklist2Matrix(self, Checklist_df, Subgraph) : 
        Matrix = list()
        for i, row in Checklist_df.iterrows() : 
            action_vec = self.Number2Vector(self._action_dict[row["text_fr_action"]], 3)
            node_idx_vec = self.Number2Vector(self.GetNodeNumber((row.p, row.u, row.t, row.m), Subgraph), 2)
            step_vec = np.concatenate([action_vec, node_idx_vec]) 
            Matrix.append(step_vec) 

        Matrix.append(np.zeros(self.target_size))
        Matrix = np.stack(Matrix, axis=0) 
        return np.pad(Matrix, ((0, self.max_out_size-Matrix.shape[0]), (0, 0)), "constant", constant_values=((0, 0), (0, 0))) 
    
    def PrintChecklistMatrix(self, Matrix) : 
        for step in range(Matrix.shape[0]) : 
            print(f"step {step}")
            print(np.reshape(Matrix[step,:], (-1, 10)).transpose())

    def NodeType2Vector(self,Node) :
        return tf.keras.utils.to_categorical(self.GetNodeType(Node), self._number_of_node_type) 
    
    
    def GetNodeNumber(self, putm, Subgraph) : 
        if isinstance(putm, tuple) : 
            p = re.sub("\s+", " ", putm[0]).strip()
            u = re.sub("\s+", " ", putm[1]).strip() 
            t = re.sub("\s+", " ", putm[2]).strip()
            m = re.sub("\s+", " ", putm[3]).strip()   
            type2 = self.GetNodeType(m)
            # if node doesn't belong to any type put it in Other Node 
            if type2 == 33 : 
                return len(Subgraph.nodes) - 1 #TRASH
            for ix, node in enumerate(Subgraph.nodes)  : 
                node = re.sub("\s+", " ", node) 
                # check if a node in the subgraph correspond to the putm
                match = re.match(rf"\s?{re.escape(p)}\s?{re.escape(u)}\s?{re.escape(t)}.*", node, re.I)    
                if match:
                    # check if the type of the node from the subgraph matches the type from the input 
                    type1 = self.GetNodeType(node)
                    if  type1 == type2  : 
                        return ix #TRASH
            
            return len(Subgraph.nodes) - 1 

            #TRASH
            #raise NotInSubgraphError(f"{' '.join([p, u, t, m])} not in the subgraph \n the nodes : {Subgraph.nodes}")
            
        else : 
            if "Other" == putm : 
                return len(Subgraph.nodes) - 1 #TRASH 
            for ix, node in enumerate(Subgraph.nodes)  : 
                if node == putm : 
                    return ix #TRASH

            return len(Subgraph.nodes) - 1
            #TRASH
            #raise NotInSubgraphError(f"{putm} not in the subgraph")

    def Node2Vector(self,Node, Subgraph)  :
        """
        Create a vector from node index and type 
        """
        return np.concatenate([self.Number2Vector(self.GetNodeNumber(Node, Subgraph), 2), self.NodeType2Vector(Node)])

    def Topology2Matrix(self,Subgraph) : 
        topo_tensor = list()
        for edge in Subgraph.edges : 
            node1_vect = self.Node2Vector(edge[0], Subgraph)
            node2_vect = self.Node2Vector(edge[1], Subgraph) 
            info_vect = np.zeros(6) 
            tensor = np.concatenate([node1_vect, node2_vect, info_vect])
            topo_tensor.append(tensor)

        # TRASH    
        # Other_node = self.Node2Vector("Other", Subgraph)
        # topo_tensor.append(np.concatenate([Other_node, Other_node, np.zeros(5)])) 
        
        return np.stack(topo_tensor, axis=0) 

    def Query2Vector(self,Node, outage, Subgraph, ON_OFF): 
        assert isinstance(outage, Outage),"outage must be an instance from Outage"
        BlankNode = np.zeros(20 + self._number_of_node_type)
        info_vect = np.zeros(6)
        info_vect[1] = 1 if ON_OFF else 0 
        info_vect[2] = 1 
        info_vect[3:] = tf.keras.utils.to_categorical(outage.value, 3) 
        return np.concatenate([self.Node2Vector(Node, Subgraph), BlankNode, info_vect])
    
    
    def ExtractSubtitleInfo(self, subtitle): 
        
        unknown_format = ["KATTE HD-T 1 12/0,4", "VERBR150 ALT G11     ", "LANGE HD-T1 12/ 0,2", " 70 788 HOUTH STALE", "CLERM 70 1/2 POSTE CÔTÉ T 1", "STRUI FEE: VCST 1", "STRUI FEE: VCST 2", "TIENE 10 COND  1  ", " 36 124 LICHT TORHO", " 70 759 BERCS ZURBN", 
                         "QUEVA 13 BLATON 1", "TURNH 15 LINK  1 R6&#8594;R7"]
        if subtitle in unknown_format : 
            raise LookupError("Unknown registered format") 

        match = re.match(r"(\w{2,5})\s+(T\s?\w{1,3}) .*", subtitle, re.I) 
        if match : 
            return (match.group(1), match.group(2), "T")

        match = re.match(r"(\w{2,5})\s+.+\s+(T\s?\w{1,3}) .*", subtitle, re.I) 
        if match : 
            return (match.group(1), match.group(2), "T" ) 

        match = re.match(r"(\w{2,5})\s+T\s?(\w{1,3})\s?\+\s?T?(\w{1,3}) .*", subtitle, re.I) 
        if match : 
            return (match.group(1), [match.group(2), match.group(3)], "T")   

        match = re.match(r"\s?\d{1,3}\s+(\d{1,3})/?(\d{1,3})?\s+(\b\w{3,5}\b)\s*-?\s*(\b\w{2,5}\b)\s*-?\s*(\b\w{2,5}\b)?\s*-?\s*(\b\w{2,5}\b)?\s?T?\s?(.*)", subtitle, re.I)
        if match : 
            if match.group(7) == "":
                return ([match.group(3),match.group(4), match.group(5), match.group(6)], [match.group(1), match.group(2)], "LI")
            else : 
                return ([match.group(3),match.group(4), match.group(5), match.group(6)], match.group(7), "T")
        match = re.match(r"\s?\d{2,3} (\d{2,3}) / (\d{2,3}) (\w{3,5}) (\w{3,5}) (\w{3,5})", subtitle, re.I)
        if match : 
            return ([match.group(3), match.group(4), match.group(5)], [match.group(1), match.group(2)], "LIs")

        match = re.match( r"\s?\d{2,3} \d{2,3} \+ \d{2,3} \w{3,5} \w{3,5} \w{3,5} \w{3,5}", subtitle, re.I)
        if match : 
            return ([match.group(3), match.group(4), match.group(5), match.group(6)], [match.group(1), match.group(2)], "LIs") 

        match = re.match(r"\d{2,3}\s+(\w{2,3}\s?\+\s?\w{2,3}) .* (\w{3,5}) (\w{3,5})", subtitle, re.I) 
        if match : 
            return ([match.group(3), match.group(4)], [match.group(1), match.group(2)],"LIs") 

        match = re.match(r"([A-Z\+0-9]{2,5})\s+T\s?([A-Z0-9]{1,3})$", subtitle, re.I) 
        if match : 
            return (match.group(1), match.group(2), "T")  

        match = re.match(r"([A-Z\+]{3,5}-[A-Z\+]{3,5}-[A-Z\+]{3,5}) \d{2,3} (\d{2,3}-\d{2,3}-\d{2,3})", subtitle, re.I)
        if match : 
            return ("-".split(match.group(1)), "-".split(match.group(2)), "LIs")

        match = re.match(r"\d{2,3} (\d{2,3}) [A-Z]\d\d \(.*\)", subtitle, re.I)
        if match : 
            return ("", match.group(1), "LI")
        
 
        raise SubtitleFormatError(f"Unknown subtitle format : {subtitle}")
    
    def FindNode(self, sectors, lines, node_type) : 

        for sector in sectors : 
            if not sector == None : 
                for line in lines : 
                    if not line == None : 
                        
                        for node in self._topo.keys() : 
                            match = re.match(r".*\s+\b(L[IK]|CK|TI)\b", node, re.I)
                            if match and node_type == "LI": 
                                match = re.match(rf"DI \d{{1,3}} ({re.escape(line)}) ((.{{3,5}})?{re.escape(sector)}(.{{0,7}})?) (L[IK]|CK|TI)", node, re.I)
                                #match = re.match(rf"DI \d{{1,3}} ({re.escape(line)}) () (L[IK]|CK|TI)", node, re.I)
                                if match : return node 

                            match = re.match(r".*\s+TFO", node, re.I)
                            if match and node_type == "TFO":

                                match = re.match(rf"({re.escape(sector)})\s?\d{{2,3}} (T\s?{re.escape(line)}) .*", node, re.I)
                                if match : 
                                    #print(f"the putm : {match.group()} | sector : {match.group(1)} | line number : {match.group(2)}")
                                    return node 
                                match = re.match(rf"({re.escape(sector)})\s?\d{{1,3}} ({re.escape(line)}).*", node, re.I) 
                                if match : 
                                    #print(f"putm : {match.group()} | sector : {match.group(1)}  | ALT number : {match.group(2)} " ) 
                                    return node 

                                match = re.match(rf"({re.escape(sector)})\s?\d{{2,3}} ({re.escape(line)}) (\d+)\s?/\s?\3 .*", node, re.I) 
                                if match : 
                                    #print(f"putm : {match.group()} | sector : {match.group(1)}  | ALT number : {match.group(2)} " ) 
                                    return node 

                                match = re.match(rf"({re.escape(sector)})\s?\d{{2,3}} ({re.escape(line)}) .*", node, re.I) 
                                if match : 
                                    #print(f"putm : {match.group()} | sector : {match.group(1)}  | ALT number : {match.group(2)} " ) 
                                    return node

                            match = re.match(r".* F?R[A-Z]*\d*[A-Z]* .*BS", node, re.I)
                            if match and "R" == node_type: 

                                match = re.match(rf"({re.escape(sector)})\s?\d{{1,3}} F?R\s?{re.escape(line)} BS", node, re.I)
                                if match : 
                                    #print(f"putm : {match.group()} | sector : {match.group(1)}  | RAIL : {match.group(2)} " ) 
                                    return node
                        raise NodeNotFoundError(f"couldn't find matching node sector {sector}, line {line}, node_type {node_type}", (sector, line, node_type))
        
    def GetSubgraph(self, node) : 
        graph = nx.Graph()
        to_add = [] 
        edges_to_add = []
        to_explore = [node]
        first_node = True
        while(len(to_explore) > 0) : 
            
            for n in to_explore :
                match = None 
                try : 
                    match = re.match(r"([\w=@]{2,5})\s?\d{1,3} (F?R\s?[A-Z]*\d*[A-Z]*) BS", n, re.I) 
                except TypeError : 
                    raise Exception(f"The node {n} is not matchable")   
                if match and not first_node: 
                    for n_to_add in self._topo[n] : 
                        if "//" in n_to_add : 
                            to_add.append(n_to_add)
                            edges_to_add.append((n, n_to_add))
                    continue 
                to_add.extend(self._topo[n]) 
                for next_n in self._topo[n] : 
                    edges_to_add.append((n, next_n))
            
            first_node = False
            graph.add_nodes_from(to_explore)
            to_explore = list(set(to_add) - set(graph.nodes))
            to_add = [] 

        graph.add_edges_from(edges_to_add) 
        graph.add_node("Other")  
        graph.add_edge("Other", "Other")  
        return graph


    def ExtractONOFF(self, s) : 
        match = re.match(r".*Remise.*", s, re.I)
        if match : 
            return False 
        match = re.match(r".*Retrait.*", s, re.I)
        if match : 
            return True 
        
        raise Exception("didn't match")

    def GetQuery(self, title_perso, subtitle, ON_OFF) : 
        match = re.match(r".*TFO\s*\d*\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "TFO")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(1)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))
            

        match = re.match(r"\(?\[?(\b\w{5}\b)?\s*\]?\s*T\s*\d+\s*\]?\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "TFO")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(2)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))

        match = re.match(r"XXXXX\s*T\s*\d+ \d*\s*/*\s*\d*\s*/*\s*\d*\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "TFO")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(1)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))

        match = re.match(r"XXXXX\s*TFO\s*\d*(\d+|[A-Z])\s*\d*\s*/*\s*\d*\s*/*\s*\d*\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "TFO")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(2)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))


        match = re.match(r".*LI\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "LI")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(1)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))

        match = re.match(r"\s*\(?(LI)?\s*\d+[\s\-\.]\d+\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "LI")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(2)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))


        match = re.match(r".*\d+ \d+ \b\w{5}\s*\w{5}\b\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "LI")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(1)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))

        match = re.match(r"\d*\s*\d*\s*/?\s*u?\s*\(?\s*\[?\s*XXXXX\s*\]?\s*\)?\s*=\s*(DV|TA|#).*", title_perso, re.I)
        if match : 
            info = self.ExtractSubtitleInfo(subtitle)
            node = self.FindNode(info[0], info[1], "R")
            
            return QueryTuple(node=node, outage=str_to_Outage(match.group(1)), subgraph=self.GetSubgraph(node), ON_OFF=self.ExtractONOFF(ON_OFF))           

        return None
    
    def CreateInputTensor(self, title_perso, subtitle, ON_OFF) : 
        query = self.GetQuery(title_perso, subtitle, ON_OFF)
        if not query == None : 
            input_tensor = np.concatenate([self.Topology2Matrix(query.subgraph), np.expand_dims(self.Query2Vector(query.node, query.outage, query.subgraph, query.ON_OFF), 0)])
            return np.pad(input_tensor, ((self.max_in_size-input_tensor.shape[0], 0), (0, 0)), "constant", constant_values=((0, 0), (0,0))), query.subgraph 
        else : 
            return None, None
    
    def PrintInputMatrix(self, Matrix): 
        for step in range(Matrix.shape[0]) : 
            print(f"step {step}")
            print("Node 1 (index): ")
            print(np.reshape(Matrix[step, :20], (-1, 10)).transpose())
            print("Node 2 (index): ")
            print(np.reshape(Matrix[step, 20+self._number_of_node_type:40+self._number_of_node_type], (-1, 10)).transpose())
            print("Node 1 (type): ") 
            print(Matrix[step, 20:20+self._number_of_node_type])
            print("Node 2 (type): ")
            print(Matrix[step, 40+self._number_of_node_type : 40+(2 * self._number_of_node_type)])
            print(f"Ans {Matrix[step,40+(2 * self._number_of_node_type)]}, Query {Matrix[step,41+(2 * self._number_of_node_type)]}, Outage {Matrix[step,42+(2 * self._number_of_node_type):45+(2 * self._number_of_node_type)]}")
            
            
            
    def CreateMaskTensor(self, mask_out_length, mask_in_length) : 
        return np.concatenate([np.zeros((mask_out_length)), np.ones((mask_in_length))], axis=0)
    
    def Create_list_cl(self, time_major=False) : 
        batch_dim = 1 if time_major else 0 
        bad_subtitle_format_list = list()
        not_in_subgraph_list = list()
        node_not_found_list = list() 
        Dataset_output_tensor = list()
        Dataset_input_tensor = list()
        Dataset_mask_tensor = list()
        current_batch_amount = 0 
        length = len(self.cl_info_df) 
        #iteration over the checklist 
        for ix, checklist_info_df in self.cl_info_df.iterrows() : 
            
            # if it doesn't concern an outage we are interested in, pass
            if not re.match(".*=\s*(DV|TA|#).*", checklist_info_df.title_perso, re.I) : 
                continue 

            # log progress    
            if ix % 100 == 0 : 
                print(f"\r{ix*100 / length} % done with {current_batch_amount} valid samples", end="")

            # gather the content of the current checklist   
            checklist_df = self.cl_df.loc[self.cl_df["proc_id"] == checklist_info_df.proc_id]

            # try to get the input tensor for the checklist
            try : 
                input_tensor, subgraph = self.CreateInputTensor(checklist_info_df.title_perso, checklist_info_df.subtitle, checklist_info_df.text_fr) 
            except NodeNotFoundError as e : 
                node_not_found_list.append((checklist_info_df.proc_id, *e.node))
                continue 
            except LookupError : 
                continue 
            except SubtitleFormatError : 
                bad_subtitle_format_list.append((checklist_info_df.subtitle, checklist_info_df.proc_id))
                continue 
            except Exception as e: 
                print(e) 
                continue

            # In case the title_perso can't be categorized, we skip this checklist
            if not isinstance(input_tensor, np.ndarray): 
                continue

            # try to create the output tensor
            try : 
                output_tensor = self.Checklist2Matrix(checklist_df, subgraph)
            except NotInSubgraphError : 
                not_in_subgraph_list.append(checklist_info_df.proc_id)
                continue    

            #the output during the graph and query phases are not needed and are zero vectors     
            discarded_answer_tensor = np.zeros((self.max_in_size, output_tensor.shape[1]))
            output_tensor = np.concatenate([discarded_answer_tensor, output_tensor])
            
            # the input vectors during answer phse
            answer_tensor = np.zeros((self.max_out_size, input_tensor.shape[1]))
            answer_tensor[:, -5] = np.ones(answer_tensor.shape[0])
            input_tensor = np.concatenate([input_tensor, answer_tensor])
            
            
            Dataset_input_tensor.append(input_tensor)
            Dataset_output_tensor.append( output_tensor) 
            current_batch_amount += 1 
            
            if len(Dataset_input_tensor) >= 3000 : 
                break
        
        dataset_tensors = Dataset_tensors(
            np.stack(Dataset_input_tensor, axis=batch_dim),
            np.stack(Dataset_output_tensor, axis=batch_dim),
            self.CreateMaskTensor(self.max_in_size, self.max_out_size))

        
        print(f"the bad subtitle list (len : {len(bad_subtitle_format_list)}) : {bad_subtitle_format_list}")
        print(f"the not in subgraph list (len : {len(not_in_subgraph_list)}) : {not_in_subgraph_list}")
        print(f"the node not found in subgraph list (len : {len(node_not_found_list)}) : {node_not_found_list}")       
        return dataset_tensors