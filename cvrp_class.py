class CVRP:
    """
   This class frames, solves, and reports on the capacitated vehicle
   routing problem, given a very specific set of inputs:
       -state, str
           Two letter abbreviation of the state to solve for.
        -dc_index, int
            The positional index of the depot from which routes originate
        -vehicle_capacity, float
            The numeric capacity of the vehicles being solved for, with respect
            to demand units
        
    kwarggs***
   
        -optimal_routing,True, bool
            Specifies if vehicle routes should be solved optimally. If true,
            a TSP-lazy constraint implementation is solved for each 
            vehicle route.
            
            If False, the route is constructed heuristically using the 
            nearest neighbor
            approach.
        -google_dr_distance, True,  bool
            Specifies if the distance matrix used in solving the 
            Clark-Wright Savings algorithm should be constructed with 
            accurate distance measures pulled from the GoogleMaps API. 
            
            Significantly increases runtime. API Key required 
            (built in for this project)
        -filename, str, 'data.csv'
            Specifies the file housing the problem formulation, 
            with the following **very** specific columns:
                -Zip - 5 Digit ZIP(TM) INT
                -City - Name STR
                -State - Abrev STR
                -Latitude - Float
                -Longitude - Float
                -geopoint - Lat+','+Long STR (found using centroid of ZIP) 
                -Demand - Demand at Geopoint NUM
                -DC_Candidate - Indicates if location can be considered for 
                depot status BIN
        - filepath, str, ''
            Indicates an alternate 
    
        Basic solution procedure:
        1. Instantiate Class
        2. Call solution method .solve
        3. examine results
        4. Change solution behavior with class attributes
        5. Solve

        """

    def __init__(self,state,dc_index,vehicle_capacity,optimal_routing = True, google_dr_dist= True, filename='data.csv',filepath='',index_col='Zip',lat_lon_column='geopoint'):
        
        # insert google maps distance matrix API key here
        self.apiKey = ‘’
        
        self.cost_per_hour = 71.8  # assuming we own trucks
        
        
        self.state = state
        self.dc_index = dc_index
        self.vehicle_capacity = vehicle_capacity

        
        self.optimal_routing = optimal_routing
        self.google_dr_dist = google_dr_dist
        self.filename = filename
        self.filepath = filepath
        self.index_col = index_col
        self.lat_lon_column = lat_lon_column
    
        
        return
    
    def refresh_data(self):
        """
        Refreshes the data for a solution run. 
        
        1. Imports fresh problem data.
        2. Filters data appropriately based on selected state (self.state)
        3. Identifies possible DCs, and selects indicated DC from positional index
        (seld.dc_index)
        """
        self.import_data()
        self.filter_data()
        self.identify_dc()
        
        if not hasattr(self,'dist_matrix'):
            print('Creating Distance Matrix')
            if self.google_dr_dist == True:
                self.distance_matrix_from_google_maps()
            else:
                self.distance_matrix_from_lat_lon()

        return
    
    
    def import_data(self):
        """
        Imports data from file specified by self.filename and self.filepath
        
        Information regarding file structure can be found in the class doc string.
        """
        import pandas as pd
        
        full_file_path = self.filename + self.filepath
        self.data = pd.read_csv(full_file_path, 
                               index_col = self.index_col)
        return
    
    
    def filter_data(self):
        '''
        Filters primary data by state.
        
        Filters the data in the self.data DataFrame so that it only includes values that match
        the self.state attribute.
        
        Creates self.filtered_data attribute. This DF object will be used extensively
        to frame and solve the problem.

        '''
        import pandas as pd

        filter_list = [self.state]

        self.filtered_data = self.data.loc[self.data['State'].isin(filter_list), :]

        return 
    
    def identify_dc(self):
        """
        Finds DCs in data, identifies the working DC.
        
        Assigns them to self.dc_candidates and self.current_dc respeictively.
        
        """
        import pandas as pd
    
        mask = self.filtered_data['DC_Candidate'] == 1
        self.dc_candidates = self.filtered_data.loc[mask].index.tolist()
        self.current_dc = self.dc_candidates[self.dc_index]
        
        return 
    
    def distance_matrix_from_lat_lon(self):
        '''
        Create a distance matrix based on latitude and longitude values contained
        in self.filtered_data. Used to compute savings in the Clark-Wright Savings procedure.

        Arguments:
        self.filtered_data: a Pandas DataFrame that includes columns with names matching
                the values provided as the self.lat_lon_column argument.

        self.lat_lon_column: name of the column that contains the latitude and longitude values
                        for each index value as a comma-separated string,
                        e.g., "lat_value,lon_value"

        '''
        import itertools
        import pandas as pd
        import numpy as np

        tuple_list = self.filtered_data[self.lat_lon_column].tolist()
        index_list = self.filtered_data.index.tolist()

        if len(tuple_list) > 1:

            tuple_permutations = list(itertools.permutations(tuple_list, 2))

            dist_df = pd.DataFrame(
                [[tup[0], tup[1]] for tup in tuple_permutations], 
                columns=["From", "To"],
            )

            dist_df[["From_Latitude", "From_Longitude"]] = dist_df["From"].str.split(
                ",", expand=True
            )
            dist_df[["To_Latitude", "To_Longitude"]] = dist_df["To"].str.split(
                ",", expand=True
            )

            def haversine(row):
                from math import radians

                lon1 = float(row["From_Longitude"])
                lat1 = float(row["From_Latitude"])
                lon2 = float(row["To_Longitude"])
                lat2 = float(row["To_Latitude"])
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                )
                c = 2 * np.arcsin(np.sqrt(a))
                km = 6367 * c
                return km

            dist_df["Distance"] = dist_df.apply(lambda row: haversine(row), axis=1)

            dist_df["Distance"] = dist_df["Distance"] * 0.621371

            self.filtered_data_index_name = self.filtered_data.index.name

            dist_df = dist_df.merge(
                self.filtered_data[self.lat_lon_column].copy().reset_index(), left_on="From", right_on= self.lat_lon_column,
            )
            dist_df = dist_df.rename(columns={self.filtered_data_index_name: "From_Index"})

            dist_df = dist_df.merge(
                self.filtered_data[self.lat_lon_column].copy().reset_index(), left_on="To", right_on=self.lat_lon_column,
            )
            dist_df = dist_df.rename(columns={self.filtered_data_index_name: "To_Index"})
            dist_df = dist_df[["From_Index", "To_Index", "Distance"]]
            dist_df = dist_df.drop_duplicates()
            dist_df = dist_df.pivot(
                index="From_Index", columns="To_Index", values="Distance"
            ).fillna(0)

            self.dist_matrix = dist_df

            return
    def distance_matrix_from_google_maps(self):
        '''
        Create a distance matrix based on latitude and longitude values contained
        in self.filtered_data.
    
        Dependencies:
        self.filtered_data: a Pandas DataFrame that includes columns with names matching
                the values provided as the self.lat_lon_column argument.
    
        self.lat_lon_column: name of the column that contains the latitude and longitude values
                        for each index value as a comma-separated string,
                        e.g., "lat_value,lon_value"
    
        '''
        import itertools
        import pandas as pd
        import numpy as np
    
        tuple_list = self.filtered_data[self.lat_lon_column].tolist()
        index_list = self.filtered_data.index.tolist()
    
        if len(tuple_list) > 1:
    
            tuple_permutations = list(itertools.permutations(tuple_list, 2))
    
            dist_df = pd.DataFrame(
                [[tup[0], tup[1]] for tup in tuple_permutations], 
                columns=["From", "To"],
            )
    
            dist_df[["From_Latitude", "From_Longitude"]] = dist_df["From"].str.split(
                ",", expand=True
            )
            dist_df[["To_Latitude", "To_Longitude"]] = dist_df["To"].str.split(
                ",", expand=True
            )
            
            def get_drive_distance(row):
                """
                Returns the driving distance between two points.
                
                API: https://developers.google.com/maps/documentation/distance-matrix/start
                """
                
                apiKey = self.apiKey
                
                from_geopoint = row["From_Latitude"] + ',' + row["From_Longitude"]
                
                to_geopoint = row["To_Latitude"] + ',' + row["To_Longitude"]
                
                origin = self.filtered_data[(self.filtered_data[self.lat_lon_column] == from_geopoint)
                                            ].index.tolist()
                
                try:
                    origin = origin[0]
                except:
                    print(f'exception at origin point {from_lat} , {from_lon}')
                    print(origin)
                    return 0
    
                destination = self.filtered_data[(self.filtered_data[self.lat_lon_column]==to_geopoint)
                                                    ].index.tolist()
                
                try:
                    destination = destination[0]
                except:
                    print(f'exception at destination point {from_lat} , {from_lon}')
                    print(destination)
                    return 0
    
    
    
                if origin == destination:
                    return 0
    
    
                import requests
                url = ('https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins={}&destinations={}&key={}'
                       .format(origin,
                               destination,
                               apiKey
                              )
                      )
                try:
                    response = requests.get(url)
                    resp_json_payload = response.json()
                    drive_distance = resp_json_payload['rows'][0]['elements'][0]['distance']['value']*0.000621371
                except:
                    print('ERROR: {}, {}'.format(origin, destination))
                    drive_distance = 99999999999
                
#                print(f'{origin} to {destination} : {drive_distance}')
                return drive_distance
    
            dist_df["Distance"] = dist_df.apply(lambda row: get_drive_distance(row), axis=1)
    
            dist_df["Distance"] = dist_df["Distance"]
    
            self.filtered_data_index_name = self.filtered_data.index.name
    
            dist_df = dist_df.merge(
                self.filtered_data[self.lat_lon_column].copy().reset_index(), left_on="From", right_on= self.lat_lon_column,
            )
            dist_df = dist_df.rename(columns={self.filtered_data_index_name: "From_Index"})
    
            dist_df = dist_df.merge(
                self.filtered_data[self.lat_lon_column].copy().reset_index(), left_on="To", right_on=self.lat_lon_column,
            )
            dist_df = dist_df.rename(columns={self.filtered_data_index_name: "To_Index"})
            dist_df = dist_df[["From_Index", "To_Index", "Distance"]]
            dist_df = dist_df.drop_duplicates()
            dist_df = dist_df.pivot(
                index="From_Index", columns="To_Index", values="Distance"
            ).fillna(0)
    
            self.dist_matrix = dist_df
    
            return
        
    def get_quadrant_info(self):
        '''
        Get quadrant and slope info in preparation for sweep algorithm.
        ****Updates the self.filtered_data attribute. 
        May interfere with other solution methods.
        Before running additional solution procedures, be sure 
        that filtered data was refreshed.

        '''
        import pandas as pd

        x_col = 'Longitude'

        y_col = 'Latitude'


        cols_to_include = self.filtered_data.columns.tolist()

        temp = self.filtered_data.copy()

        dc_x = temp.at[self.current_dc, x_col]
        dc_y = temp.at[self.current_dc, y_col]

        temp["x_norm"] = temp[x_col] - dc_x
        temp["y_norm"] = temp[y_col] - dc_y

        temp["Distance"] = 0
        temp["Slope"] = 0
        temp["Quadrant"] = 0

        for index, row in temp.iterrows():

            if row["x_norm"] != 0:
                temp.loc[index, "Slope"] = row["y_norm"] / row["x_norm"]
            else:
                temp.loc[index, "Slope"] = 0
            if (row["y_norm"] >= 0) & (row["x_norm"] >= 0):
                temp.loc[index, "Quadrant"] = 1
            elif (row["y_norm"] >= 0) & (row["x_norm"] < 0):
                temp.loc[index, "Quadrant"] = 2
            elif (row["y_norm"] < 0) & (row["x_norm"] < 0):
                temp.loc[index, "Quadrant"] = 3
            else:
                temp.loc[index, "Quadrant"] = 4

        self.filtered_data = temp[cols_to_include + ["Slope","Quadrant",]]   
        return    
    
    def sweep_procedure(self):   
        # sort data by quadrant and slope in ascending order
        self.filtered_data = self.filtered_data.sort_values(by=["Quadrant", "Slope"])

        # initialize list of unvisited nodes
        not_visited = self.filtered_data.index.tolist()
        
        self.route_descriptions = {}
        
        # initialize current vehicle to 1
        current_vehicle = 1
        
        self.route_descriptions[current_vehicle] = {'route':[],'capacity_used':0,'rem_capacity':0}

        
        # initialize remaining capacity to vehicle capacity
        remaining_capacity = self.vehicle_capacity

        # while there are unvisited nodes
        while not_visited:

            # set the current customer to the first unvisited node
            customer = not_visited[0]

            # set the current demand to that associated with the first unvisited node
            customer_demand = self.filtered_data.loc[customer, "Demand"]

            # if the current demand is less than or equal to the remaining capacity
            if customer_demand <= remaining_capacity:

                # assign customer to the current vehicle
                self.filtered_data.loc[customer, "Vehicle"] = current_vehicle
                self.route_descriptions[current_vehicle]['route'].append(customer)


            # else if the current demand is greater than the remaining capacity
            else:

                # add a new vehicle (increment current vehicle)
                current_vehicle += 1
                self.route_descriptions[current_vehicle] = {'route':[],'capacity_used':0,'rem_capacity':0}
                self.route_descriptions[current_vehicle]['route'].append(customer)

                # reset remaining capacity to vehicle capacity
                remaining_capacity = self.vehicle_capacity

                # assign customer to the current vehicle
                self.filtered_data.loc[customer, "Vehicle"] = current_vehicle

            # remove customer from list of unvisited nodes
            not_visited.remove(customer)

            # reduce remaining capacity by customer's demand
            remaining_capacity -= customer_demand
            self.route_descriptions[current_vehicle]['rem_capacity'] = remaining_capacity
    
            self.route_descriptions[current_vehicle]['capacity_used'] += customer_demand

        return
    
    def cw_savings_procedure(self): 
        import pandas as pd
        import itertools
        
        pd.options.mode.chained_assignment = None

        
        all_zips = self.filtered_data.index.tolist()

        zip_pairs = [
            tup for tup in itertools.combinations(all_zips, 2) if self.current_dc not in tup
        ]

        savings = []
        for start_zip, end_zip in zip_pairs:
            pair_savings = (
                self.dist_matrix.loc[self.current_dc, start_zip]
                + self.dist_matrix.loc[end_zip, self.current_dc]
                - self.dist_matrix.loc[start_zip, end_zip]
            )
            savings.append([start_zip, end_zip, pair_savings])
        savings = pd.DataFrame(savings, columns=["From", "To", "Savings"])

        savings.pivot(index="From", columns="To", values="Savings")

        savings = savings.sort_values(by="Savings", ascending=False).reset_index(drop=True)

        savings.index = [x for x in zip(savings.pop('From'), savings.pop('To'))]

        # create dictionary to store route for each location
        route_assignments = {zip_code: zip_index for zip_index, zip_code in enumerate(all_zips)}

        # create dictionary to store information for each route
        route_descriptions = {}
        for zip_code, route in route_assignments.items():
            if route in route_descriptions.keys():
                route_descriptions[route]["route"].append(zip_code)
                route_descriptions[route]["rem_capacity"] -= self.filtered_data.at[
                    zip_code, "Demand"
                ]
            else:
                route_descriptions[route] = {
                    "route": [zip_code],
                    "capacity_used": self.filtered_data.at[zip_code, "Demand"],
                    "rem_capacity": (self.vehicle_capacity - self.filtered_data.at[zip_code, "Demand"]),
                }

        # for all origin destination pairs in the savings data
        for from_index, to_index in savings.index:

            # get the current route for the "from" and "to" locations
            from_route = route_assignments[from_index]
            to_route = route_assignments[to_index]

            # if the current route for the "from" and "to" locations differ
            if from_route != to_route:

                # if the current from rout can accommodate all stops on the to route
                if (route_descriptions[from_route]["rem_capacity"]
                    >= route_descriptions[to_route]["capacity_used"]):

                    # update the route assignment for all stops in the route for 
                    # the "to" location
                    for stop in route_descriptions[to_route]["route"]:
                        route_assignments[stop] = from_route

                    # add the stops in the route for the "to" location to
                    # the route for the #from location
                    route_descriptions[from_route]["route"].extend(route_descriptions[to_route]["route"])

                    # update capacities for the modified route for the "from" location
                    route_descriptions[from_route]["rem_capacity"] -= route_descriptions[to_route]["capacity_used"]
                    route_descriptions[from_route]["capacity_used"] += route_descriptions[to_route]["capacity_used"]

                    # remove the route for the "to" location
                    route_descriptions.pop(to_route)


        # get DC route
        dc_route = route_assignments[self.current_dc]

        # for all routes
        for route in route_descriptions.keys():

            # if the current from rout can accommodate the DC location
            if ((route_descriptions[route]["rem_capacity"] 
                 >= self.filtered_data.at[self.current_dc, "Demand"]) 
                & (route != dc_route)):

                # update the current route to incldue the DC
                route_descriptions[route]["route"].append(self.current_dc)

                # update the DC's route assignment
                route_assignments[self.current_dc] = route

                # update the current routes capacities
                route_descriptions[route]["rem_capacity"] -= self.filtered_data.at[self.current_dc, "Demand"]
                route_descriptions[route]["capacity_used"] += self.filtered_data.at[self.current_dc, "Demand"]

                # remove the DC's previous route
                route_descriptions.pop(dc_route)

                # exit the for loop
                break

#        vehicle = 1
#        for val in route_descriptions.values():
#            self.route_descriptions[vehicle] = val
#            vehicle += 1
        
        temp_dict = {}
        itt = 1
        for vehicle in route_descriptions.keys():
            temp_dict[itt] = route_descriptions[vehicle]
            itt += 1
        
        self.route_descriptions = temp_dict
        
        for route_index, route in enumerate(route_descriptions.keys()):
            self.filtered_data.loc[route_descriptions[route]["route"], "Vehicle"] = route_index + 1

        return    
    
    def order_routes(self):
        def get_nn_tour(vehicle, verbose = False):
            start_city = self.current_dc
            
            initial_route = self.route_descriptions[vehicle]['route']
            
            if start_city in initial_route:
                pass
            else:
                initial_route.append(start_city)
            
            
            data = self.dist_matrix.loc[initial_route,initial_route]
            
            
            
            NN_tour = [start_city]
            
            unvisited = data.index.tolist()
            
            
            unvisited.remove(start_city)
            
            itt = 1
            while unvisited:

                next_city = data.loc[start_city, unvisited].idxmin()
                NN_tour.append(next_city)
                unvisited.remove(next_city)
                
                start_city = next_city
                
            self.route_descriptions[vehicle]['route'] = NN_tour
            
            return 
            
        def get_optimal_tour(vehicle, warmstart = None, verbose = False, time_limit = None):
            '''
            This function solves the travelling salesman problem for the problem defined
            by a given distance matrix. The function expects the following arguments:
            
            dist_df: a Pandas DataFrame that specifies the distance between origin-destination (OD) pairs.
            All possible values for the origin and desinatiion should be included in the index 
            columns of the DataFrame. The value at the intersection of the labels for an OD pair  
            is the distance from the origin (row label) to the destination (column label).
            
            warmstart: a list specifying a tour to use as an initial solution (optional)
            
            verbose: a flag that can be set to True or False to limit the amount of information
            from Gurobi that is written to the screen (optional)
            
            time_limit: the maximum amount of time that is allowed for Gurobi to attempt to 
            solve the problem (optional)
            
            '''
            
            import numpy as np
            import pandas as pd
            import gurobipy as grp
            
            start_city = self.current_dc
        
            initial_route = self.route_descriptions[vehicle]['route']
        
            if start_city in initial_route:
                pass
            else:
                initial_route.append(start_city)
        
        
            dist_df = self.dist_matrix.loc[initial_route,initial_route]
            
            
            
            # Callback - use lazy constraints to eliminate sub-tours
            def subtourelim(model, where):
                if where == grp.GRB.Callback.MIPSOL:
                    # make a list of edges selected in the solution
                    vals = model.cbGetSolution(model._vars)
                    selected = grp.tuplelist((i,j) for i,j in model._vars.keys() if vals[i,j] > 0.5)
                    # find the shortest cycle in the selected edge list
                    tour = subtour(selected)
                    if len(tour) < n:
                        tour_sum = grp.LinExpr()
                        for tour_index, tour_stop in enumerate(tour):
                            if tour_index < len(tour)-1:
                                tour_sum += model._vars[tour[tour_index], tour[tour_index+1]]
                            else:
                                tour_sum += model._vars[tour[tour_index], tour[0]]
                        model.cbLazy(tour_sum <= len(tour)-1)
        
        
            # Given a list of tuples containing the tour edges, find the shortest subtour
            def subtour(edges):
                unvisited = list(dist_df.index)
                cycle = unvisited + [unvisited[0]] # initial length has 1 more city
                while unvisited: # true if list is non-empty
                    thiscycle = []
                    neighbors = unvisited
                    while neighbors:
                        current = neighbors[0]
                        thiscycle.append(current)
                        unvisited.remove(current)
                        neighbors = [j for i,j in edges.select(current,'*') if j in unvisited]  
                    if len(cycle) > len(thiscycle):
                        cycle = thiscycle
                return cycle
        
        
            # Dictionary of Euclidean distance between each pair of points
            dist = {(i,j): dist_df.loc[i,j] for i in dist_df.index for j in dist_df.index if i != j}
        
            n = len(dist_df.index)
        
            m = grp.Model()
            if not verbose:
                m.params.OutputFlag = 0
                
            if time_limit:
                m.params.TimeLimit = time_limit
        
            # Create variables
            trav = m.addVars(dist.keys(), 
                             obj = dist, 
                             vtype = grp.GRB.BINARY, 
                             name='e')
            
            if warmstart:
                for warmstart_index, warmstart_stop in enumerate(warmstart):
                    if warmstart_index < len(warmstart)-1:
                        trav[warmstart[warmstart_index], warmstart[warmstart_index+1]].Start = 1
                    else:
                        trav[warmstart[warmstart_index], warmstart[0]].Start = 1
        
            for i in dist_df.index:
                j_sum = grp.LinExpr()
                for j in dist_df.index:
                    if j != i:
                        j_sum += trav[i,j]
                m.addConstr(j_sum == 1)
        
            for j in dist_df.index:
                i_sum = grp.LinExpr()
                for i in dist_df.index:
                    if j != i:
                        i_sum += trav[i,j]
                m.addConstr(i_sum == 1)
        
            # Optimize model
            m._vars = trav
            m.params.Heuristics = 0
            m.params.LazyConstraints = 1
        
            m.optimize(subtourelim)
        
            try:
                vals = m.getAttr('X', trav)
                selected = grp.tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)
        
                tour = subtour(selected)
                # assert len(tour) == n
        
                if verbose:
                    print('')
                    print('Optimal tour: %s' % str(tour))
                    print('Optimal cost: %g' % m.objVal)
                
                
                
                
                self.route_descriptions[vehicle]['route'] = tour
                
                return 
            
            except:
                print('No solution found')        
        
        
        if self.optimal_routing == True:
            route_method = get_optimal_tour
        else:
            route_method = get_nn_tour
        
        for vehicle in self.route_descriptions.keys():
            route_method(vehicle)
            
            tour = self.route_descriptions[vehicle]['route']
            
            tour = self.route_descriptions[vehicle]['route']
            i = tour.index(self.current_dc)
            return_home = [self.current_dc]
            
            self.route_descriptions[vehicle]['route'] = tour[i:] + tour[:i] + return_home
            
            tour = self.route_descriptions[vehicle]['route']

        
        
        
        
        return
        
    
    def get_vrp_plot(self):
        '''
        Returns a plot for a VRP solution. 

        Arguments:
        self.filtered_data: a Pandas DataFrame that specifies the x and y coordinate data 
                 and the vehicle assignments for the instance

        x_col: column containing x-coordinate values (note that higher values
                should represent values farther to the right when drawn on a 
                2D plane)

        x_col: column containing x-coordinate values (note that higher values
                should represent values farther to the top when drawn on a 
                2D plane)

        self.current_dc: the index corresponding to the depot

        vehicle_col: column containing vehicle assignments

        ax: a matplotlib axis

        '''

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
            
        
#        get state info
        
        state_df = pd.read_csv('Border_Geopoints.csv')
        state_df = state_df[self.state].dropna().tolist()
        
        x_list = []
        y_list = []
        
        for tup in state_df:
            x,y = tup.split(',')
            x_list.append(float(x))
            y_list.append(float(y))
        
        
        x_col = 'Longitude'
        y_col = 'Latitude'
        vehicle_col = 'Vehicle'

        fig, ax = plt.subplots(1,1, figsize=(9, 12))

        ax.plot(x_list,y_list)
        
        
        ax.scatter(
            self.filtered_data[x_col], 
            self.filtered_data[y_col], 
            c=self.filtered_data[vehicle_col], 
            edgecolor="k", 
            s=50,
        )

        ax.scatter(
            self.filtered_data.loc[self.current_dc, x_col],
            self.filtered_data.loc[self.current_dc, y_col],
            edgecolor="k",
            marker="*",
            c="red",
            s=1000,
        )

        ax.axvline(
            self.filtered_data.loc[self.current_dc, x_col], 
            color="k",
        )
        ax.axhline(
            self.filtered_data.loc[self.current_dc, y_col], 
            color="k",
        )

        ax.set_title(f"{self.recent_method} Solution for DC at {self.current_dc} in {self.state}")

        
        return plt.show()
    
    def solve(self,method,verbose = True):
        """
        Solves the CVRP based on user params.
        
        Method- 'sweep' or 'clark-wright savings'
        Verbose- T/F, If true, output printed to console
        
        Dependencies:
            -self.refresh_data() to ensure data integrety
            -solution methods to produce output
        
        Output:
            - self.route_descriptions with route information
        """
        self.refresh_data()
        
        if method.lower() == 'sweep':
            self.recent_method = 'Sweep Heuristic'
            
            self.get_quadrant_info()
            self.sweep_procedure()
            
            
        
        elif method.lower() == 'clark-wright':
            self.recent_method = 'Clark-Wright Savings'
            self.cw_savings_procedure()
        else:
            print(f'Invalid solution method: {method}\nPlease select from one of the following:\nClark-Wright Savings, Sweep')
            return
         
        self.order_routes()
        self.calculate_route_time()
        self.calculate_cost()
        
        if verbose:
            return self.summarize_solution() , self.get_vrp_plot()   

        return
    
    def get_drive_time(self, origin, destination):
        """
        Returns the driving time between two points.
        
        API: https://developers.google.com/maps/documentation/distance-matrix/start
        """
        import requests
        url = ('https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins={}&destinations={}&key={}'
               .format(origin,
                       destination,
                       self.apiKey
                      )
              )
        try:
            response = requests.get(url)
            resp_json_payload = response.json()
            drive_time = resp_json_payload['rows'][0]['elements'][0]['duration']['value'] / 60 / 60
        except:
            print('ERROR: {}, {}'.format(origin, destination))
            drive_time = 0
        return drive_time
    
    def calculate_route_time(self,verbose=False):
        
        temp_dictionary = {}
        
        for vehicle in self.route_descriptions.keys():
            route = self.route_descriptions[vehicle]['route']
    
            google_list = []
    
            google_list.append((self.current_dc,route[0]))
    
            for itt in range(len(route)-1):
                google_list.append((route[itt],route[itt+1]))
    
            google_list.append((route[-1],self.current_dc))
    
            temp_time = 0
            for segment in google_list:
                origin = segment[0]
                destination = segment[1]
                temp_time += self.get_drive_time(origin,destination)
    
            temp_dictionary[vehicle] = temp_time
            if verbose == True:
                print(f'Vehicle {vehicle} succesfully calculated. Drive Time {temp_time} hrs')
            
        for vehicle in temp_dictionary.keys():
            self.route_descriptions[vehicle]['drive_time'] = temp_dictionary[vehicle]
        
        return
    
    def calculate_cost(self):
        """ 
        Calculates total cost of the implemented solution
        
        Returns:
        Cost of the route
        Output:
        The total cost of the Drive time for each route
        
        """
        
        self.total_time = 0
        
        self.num_vehicles = len(self.route_descriptions.keys())
        
        for vehicle in  self.route_descriptions.keys():
            self.total_time +=  self.route_descriptions[vehicle]['drive_time']
            
        self.total_cost = self.total_time*self.cost_per_hour
    
        return 
    
    def summarize_solution(self):
        """
        Summarizes the solution.
        
        Arguments:
        Dependancies
        Outputs:
        For each vehicle
            Vehicle - Number of the vehicle used
            Route - Series of ZIP codes from start to finish of the route
            Drive Time - Amount of time to complete the route from start to finish provided by the google API
            Capacity Used - Amount of space used compared to total capacity
            Capacity Percentage - Capacity used represented as a percentage
        For each DC
            Total Cost - Total cost calculated using sum(drive time(s))* marginal operating cost of 18 wheeler in 2018
            Total Time - Sum(drive time) for all routes
        """         
        ls = 70
        num_vehicles = len(self.route_descriptions)
        
        title = f"{self.recent_method} , Opt: {self.optimal_routing} for DC {self.current_dc} in {self.state}"
        
        print(title.center(ls,'*'))
        print(f'Vehicles Used: {num_vehicles}')
        
        for vehicle, summary in self.route_descriptions.items():
            route = summary['route']
            capacity_used = summary['capacity_used']
            capacity_percentage = capacity_used / self.vehicle_capacity
            capacity_percentage = '{:.1%}'.format(capacity_percentage)
            drive_time = self.route_descriptions[vehicle]['drive_time']

            print(f'Vehicle {vehicle}'.center(ls,'-'))
            print(f'Route: {route}')
            print(f'Drive time: {round(drive_time,2)} hrs')
            print(f'Capacity Used: {capacity_used} / {self.vehicle_capacity}')

            print(f'Utilization: {capacity_percentage}')
        print(''.center(ls,'-'))
        print(f'Total Cost: ${round(self.total_cost,2)}')
        print(f'Total Time: {round(self.total_time,2)} hrs')
        print(''.center(ls,'-'))
        
            
            
            
            
            