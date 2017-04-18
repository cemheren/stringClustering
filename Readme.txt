To create a single exe:
pyinstaller main.py --hidden-import sklearn.neighbors.typedefs --onefile

To build the docker image: 
docker build -t governanceint.azurecr.io/string-clusterer .

To run the docker image (for console):
docker run -it governanceint.azurecr.io/string-clusterer python /src/console.py test.txt --verbose

To run the docker image (for server):
docker run -d -p 80:5000 governanceint.azurecr.io/string-clusterer python /src/server.py
the server runs on your local default docker vm (find ip address using ipconfig)

To push a new image to the container registry: 
1) login to the container registry: docker login governanceint.azurecr.io -u GovernanceINT -p <password>
2) docker push governanceint.azurecr.io/string-clusterer

To pull from the container registry: 
1) login.
2) docker pull governanceint.azurecr.io/string-clusterer

