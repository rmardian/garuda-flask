import httplib2
import json

http = httplib2.Http()
url = "http://localhost:9000/api"

def resolveSearch(data, user, authHeader):

	if authHeader == None:
		return ("Not authorized!", None)

	filtered_results = []

	if 'biodesignId' in data:
		if data['biodesignId'] != None and data['biodesignId'] != "":

			response, content = http.request(url + "/device/" + data['biodesignId'], 'GET', headers={'Content-Type': 'application/json', 'Authorization': authHeader})

			output = None
			if response.status == 200:
				output = json.loads(content.decode("utf-8"))
				parsedResult = parseResult(output[0], None)
				if parsedResult != None:
					filtered_results.append(parsedResult)

			print(len(filtered_results))
			print(filtered_results)

			return ("Found " + str(len(filtered_results)) + " match results!!!", filtered_results)

	elif 'sequence' in data:
		if data['sequence'] != None and data['sequence'] != "":

			#to be implemented
			return ("Search with BLAST", None)

	else:

		skip = False
		role = None

		if 'role' in data:
			if data['role'] != None and data['role'] != "":

				role = data['role']
				data.pop('role', None)		#to avoid clotho error when role is present

				if 'type' in data:
					if data['type'] == "DEVICE":
						skip = True
						return "Warning: no device should have a role!"

				data['type'] = "PART"

		if not skip:

			response, content = http.request(url + "/bioDesign/search", 'PUT', json.dumps(data), headers={'Content-Type': 'application/json', 'Authorization': authHeader})

			output = None
			if response.status == 200:
				output = json.loads(content.decode("utf-8"))["data"]

				for i in range(len(output)):

					parsedResult = parseResult(output[i], role)

					if parsedResult != None:
						filtered_results.append(parsedResult)

			return ("Found " + str(len(filtered_results)) + " match results!!!", filtered_results)

	return ("Something wrong! Please check your query!", None)


def parseResult(obj, query_role):

	parts = []
	json = {}

	if isinstance(obj, dict):

		json['_id'] = obj['_id']
		json['name'] = obj['name']
		json['type'] = obj['type']
		json['userId'] = obj['userId']

		print(obj['_id'], obj['name'])

		if obj['type'] == "DEVICE":
			subdesigns = obj['subdesigns']
			for i in range(len(subdesigns)):
				parts = parseDevice(parts, subdesigns[i])

		elif obj['type'] == "PART":
			modules = obj['modules']
			if len(modules) > 0:
				role = modules[0]['role']
				if role == "PROMOTER":
					role = "Promoter"
				elif role == "GENE":
					role = "CDS"
				elif role == "TERMINATOR":
					role = "Terminator"
				parts.append({'name' : obj['name'], 'role' : role})
			else:
				parts.append({'name' : obj['name'], 'role' : ''})

		json['parts'] = parts

		return json

	elif isinstance(obj, list):

		entry = obj[0]

		json['_id'] = entry['_id']
		json['name'] = entry['name']
		json['type'] = entry['type']
		json['userId'] = entry['userId']

		print(entry['_id'], entry['name'])

		if entry['type'] == "PART":
			modules = entry['modules']
			if len(modules) > 0:
				role = modules[0]['role']

				if query_role != None and role != query_role:
					return None

				if role == "PROMOTER":
					role = "Promoter"
				elif role == "GENE":
					role = "CDS"
				elif role == "TERMINATOR":
					role = "Terminator"
				parts.append({'name': entry['name'], 'role': role})
			else:
				parts.append({'name': entry['name'], 'role': ''})

		json['parts'] = parts

	return json

def parseDevice(parts, obj):

	if obj['type'] == "DEVICE":
		subdesigns = obj['subdesigns']
		for i in range(len(subdesigns)):
			parts = parseDevice(parts, subdesigns[i])

	elif obj['type'] == "PART":
		modules = obj['modules']
		if len(modules) > 0:
			role = modules[0]['role']
			if role == "PROMOTER":
				role = "Promoter"
			elif role == "GENE":
				role = "CDS"
			elif role == "TERMINATOR":
				role = "Terminator"
			parts.append({'name': obj['name'], 'role': role})
		else:
			parts.append({'name': obj['name'], 'role': ''})

	return parts