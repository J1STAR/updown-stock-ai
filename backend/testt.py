from sklearn.preprocessing import MinMaxScaler


sc = MinMaxScaler()

input = [[5], [6], [7], [8]]

input = sc.fit_transform(input)

print(input)

input = sc.inverse_transform(input)

print(input)