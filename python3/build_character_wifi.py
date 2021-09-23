import urllib.request

def call(con):
    #url = "http://172.20.10.6/" + str(con)
    url = "http://172.20.10.7/"  + str(con) #simon's iphone
    n = urllib.request.urlopen(url).read() # get the raw html data in bytes (sends request and warn our esp8266)
    n = n.decode("utf-8") # convert raw html bytes format to string :3

    # data = n
    data = n.split() 			#<optional> split datas we got. (if you programmed it to send more than one value) It splits them into seperate list elements.
    #data = list(map(int, data)) #<optional> turn datas to integers, now all list elements are integers.
    return data

# Example usage
while True:
    con = input("Give input: ")

    data = call(con)
    #print("Your data(s) which we received from arduino: "+data)

    f = open("node11.txt", "w")
    fullstr=' '.join(data)
    f.write(fullstr)
    f.close()

    f = open("node11.txt", "r")
    print(f.read())
