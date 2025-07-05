def ProcessData(data):
 for i in range(len(data)):
  if data[i] == 0: data[i] = 1
 print("processed")
 return data

d = [0,1,2,0,4]
ProcessData(d)
