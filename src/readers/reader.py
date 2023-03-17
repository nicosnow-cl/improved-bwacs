class ReaderCVRPLIB:
    def __init__(self, file_name, path = 'datasets', max_nodes = 99999):
        import os
        import math
        
        self.path = path
        self.file_name = file_name
        self.max_nodes = max_nodes
        self.os = os
        self.math = math
    
    def read(self):
        if self.file_name:
            self.path = self.os.getcwd() + '/' + self.path + '/'
            try:
                with open(self.path + self.file_name + '.vrp', 'r') as f:
                    del_index = []
                    data = {}                    
                    temp_file = f.read().split('\n') # Guardamos cada linea de el archivo dentro de una variable temporal                   
                    
                    for i, line in enumerate(temp_file):
                        temp_file[i] = line.replace('\t', ' ')
                    
                    for line in temp_file:                        
                        temp_line = line.split(':')
                        temp_line[0] = temp_line[0].replace(' ', '')
                        if temp_line[0] == 'NODE_COORD_SECTION':
                            break
                        else:                                                  
                            data[temp_line[0]] = temp_line[1]
                            del_index.append(line)
                            
                    for i in del_index:
                        temp_file.remove(i)                  
                    
                    del_index.clear()
                    for line in temp_file:
                        if 'NODE_COORD_SECTION' in line:
                            data['NODE_COORD_SECTION'] = {}
                            del_index.append(line)
                            idx = 0
                        elif 'DEMAND_SECTION' in line:
                            break
                        else:
                            temp_line = line.split()
                            data['NODE_COORD_SECTION'][idx] = [int(float(temp_line[1])), int(float(temp_line[2]))]
                            del_index.append(line)
                            idx += 1
                    
                    for i in del_index:
                        temp_file.remove(i)

                    del_index.clear()
                    for line in temp_file:
                        if 'DEMAND_SECTION' in line:
                            data['DEMAND_SECTION'] = {}
                            del_index.append(line)
                            idx = 0
                        elif 'EOF' in line:
                            break
                        else:
                            temp_line = line.split()
                            temp_line[1] = temp_line[1].replace('.', ',') 
                            data['DEMAND_SECTION'][idx] = int(temp_line[1])
                            del_index.append(line)
                            idx += 1
                    
                    depot = list(data['NODE_COORD_SECTION'].keys())[:1][0]
                    clients = list(data['NODE_COORD_SECTION'].keys())[1:self.max_nodes]
                    loc_x = {node: data['NODE_COORD_SECTION'][node][0] for node in [depot] + clients}
                    loc_y = {node: data['NODE_COORD_SECTION'][node][1] for node in [depot] + clients}
                    demands = {node: data['DEMAND_SECTION'][node] for node in [depot] + clients}
                    total_demand = sum(demands.values())
                    vehicle_capacity = int(data['CAPACITY'])
                    k = self.math.ceil(total_demand/vehicle_capacity)
                    tightness_ratio = total_demand/(k * vehicle_capacity)

                    return depot, clients, loc_x, loc_y, demands, total_demand, vehicle_capacity, k, tightness_ratio

            except Exception as e: 
                print('Ocurrio un ERROR: ' + str(e))
        else:
            print('Debe ingresar el nombre del archivo CVRPLIB como parametro. Es necesario que en el PATH exista tanto el archivo .vrp.')