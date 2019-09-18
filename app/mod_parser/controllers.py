import httplib2
import pandas as pd

from app.mod_clotho.controllers import createPart, createDevice, deleteDevice

http = httplib2.Http()
url = "http://localhost:9000/api"

ALLOWED_EXTENSIONS = set(['xlsx', 'xls', 'csv'])

def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resolveImport(filename, user, authHeader):

	if user == "robwarden":
		message = robParser(filename, user, authHeader)
	#elif user == "mardian":
	#	message = mardianParser()
	#else:
	#	message = generalParser()

	return message

def robParser(filename, user, authHeader):

	xls = pd.ExcelFile(filename)
	sheets = xls.sheet_names

	parts_dict = {}
	lvl_1_dict = {}

	for i in range(len(sheets)):

		df = pd.read_excel(filename, sheet_name=sheets[i])

		if sheets[i] == "Rules":

			print("===Populate Parts===")

			parts = []
			parts.append([(x, "RIBOZYME") for x in set(df["Ribozyme"]) if type(x) == str])
			parts.append([(x, "TERMINATOR") for x in set(df["Terminator"]) if type(x) == str])
			parts.append([(x, "GENE")  for x in set(df["Gene"]) if type(x) == str])
			parts.append([(x, "RBS") for x in set(df["RBS"]) if type(x) == str])
			parts.append([(x, "PROMOTER") for x in set(df["Promoter"]) if type(x) == str and x != "end"])

			parts = [item for sublist in parts for item in sublist]

			for part in parts:
				payloads = {'name': part[0],
						'displayId': part[0],
						'role': part[1]
						}
				partId = createPart(payloads, user, authHeader)
				parts_dict[payloads['displayId']] = partId

			print(parts_dict)

		elif sheets[i] == "Parts":

			print("===Populate Construct Level 1===")

			poscispart = [x for x in df["PosCis_Part"].tolist() if type(x) == str and x != "H2O"]
			promoter = [x for x in df["Promoter"].tolist() if type(x) == str and x != "H2O"]
			ribozyme = [x for x in df["Ribozyme"].tolist() if type(x) == str and x != "H2O"]
			rbs = [x for x in df["RBS"].tolist() if type(x) == str and x != "H2O"]
			enzyme = [x for x in df["Enzyme"].tolist() if type(x) == str and x != "H2O"]
			terminator = [x for x in df["Terminator"].tolist() if type(x) == str and x != "H2O"]

			lvl_1 = [list(item) for item in zip(poscispart, promoter, ribozyme, rbs, enzyme, terminator)]

			for construct in lvl_1:
				payloads = {'name': construct[0],
					'displayId': construct[0]
				}
				partIds = ""
				for field in construct[1:]:
					partIds += '"' + parts_dict[field] + '",'
				partIds = partIds[:-1]
				payloads['partIds'] = "[" + partIds + "]"
				payloads['createSeqFromParts'] = "True"

				lvl_1_Id = createDevice(payloads, user, authHeader)
				lvl_1_dict[payloads['displayId']] = lvl_1_Id

			print(lvl_1_dict)

		if sheets[i] == "Enumerated Constructs":

			print("===Populate Construct Level 2===")

			lvl_2 = [list(item) for item in zip(df["Order"].tolist(),
												df["Positional_Cistron_1"].tolist(), df["Positional_Cistron_2"].tolist(), df["Positional_Cistron_3"].tolist(),
												df["Positional_Cistron_4"].tolist(), df["Positional_Cistron_5"].tolist(), df["Positional_Cistron_6"].tolist())]

			for construct in lvl_2:
				payloads = {'name': str(construct[0]),
					'displayId': str(construct[0])
				}
				partIds = ""
				for field in construct[1:]:
					if field != "H2O":
						partIds += '"' + lvl_1_dict[field] + '",'
				partIds = partIds[:-1]
				payloads['partIds'] = "[" + partIds + "]"
				payloads['createSeqFromParts'] = "True"

				createDevice(payloads, user, authHeader)

		#elif sheets[i] == "Final Strains":

		#elif sheets[i] == "Assemblies":

	return "Import successful!"