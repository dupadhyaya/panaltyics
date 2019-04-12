
# portfolio.py

total = 0.0
f     = open("portfolio.dat","r")

for line in f:
    fields = line.split()
    name   = fields[0]
    shares = int(fields[1])
    price  = float(fields[2])
    total += shares*price
    print("%-10s %8d %10.2f" % (name,shares,price))

f.close()
print("Total", total)