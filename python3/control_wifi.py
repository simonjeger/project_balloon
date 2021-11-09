import urllib.request

def call(url):
    n = urllib.request.urlopen(url).read() # get the raw html data in bytes (sends request and warn our esp8266)
    n = n.decode("utf-8") # convert raw html bytes format to string :3

    # data = n
    data = n.split() 			#<optional> split datas we got. (if you programmed it to send more than one value) It splits them into seperate list elements.
    #data = list(map(int, data)) #<optional> turn datas to integers, now all list elements are integers.
    return data

# Example usage
while True:
    con = input("Give input for fan 0: ") #0.79
    url = "http://192.168.0.163/"  + str(con)
    data = call(url)

    con = input("Give input for fan 1: ")
    url = "http://192.168.0.111/"  + str(con)
    data = call(url)
