import socket
import pandas as pd

# 產生socket物件
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

# 绑定ip和port
server.bind(('localhost', 9999))
# 監聽绑定的port
server.listen(1)

dataset_train = pd.read_csv('./rds_cpu_utilization_cc0c53.csv')
stream_data = dataset_train.values
print(stream_data)
while 1:
        # 為了識別方便起見，在螢幕上印出"I'm waiting to connect..."
        print("I'm waiting to connect...")
        conn,addr = server.accept()
        # 印出"連接成功"
        print("Connect success! Connection is from %s " % addr[0])
        # 列印要發送的資料
        print('Sending data...')
        conn.send('I love hadoop I love spark I love NUU Spark Spark Spark Hadoop Hadoop'.encode())
        conn.close()
        print('Connection is broken.') 