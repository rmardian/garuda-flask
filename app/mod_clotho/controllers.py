import httplib2
import json

http = httplib2.Http()
url = "http://localhost:9000/api"

def createPart(data, user, authHeader):

	if authHeader == None:
		return "Not authorized!"

	response, content = http.request(url + "/part", 'POST', json.dumps(data), headers={'Content-Type': 'application/json', 'Authorization': authHeader})

	output = None
	if response.status == 200:
		output = content.decode("utf-8")

	print("created a part with ID", output)

	return output

def createDevice(data, user, authHeader):

	if authHeader == None:
		return "Not authorized!"

	response, content = http.request(url + "/device", 'POST', json.dumps(data), headers={'Content-Type': 'application/json', 'Authorization': authHeader})

	output = None
	if response.status == 200:
		output = content.decode("utf-8")

	print("created a device with ID", output)

	return output


def deleteDevice(id, user, authHeader):

	if authHeader == None:
		return "Not authorized!"

	response, content = http.request(url + "/device/" + id, 'DELETE', headers={'Content-Type': 'application/json', 'Authorization': authHeader})

	output = None

	print(response)
	print(content)

	if response.status == 200:
		output = content.decode("utf-8")

	print("delete a device with ID", output)

	return output