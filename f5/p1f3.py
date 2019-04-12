#Write a program that reads the stocks in "portfolio.dat" into memory.
#Alphabetize the stocks and print a report.
## Calculate the initial value of the portfolio.
# portfolio.py
#http://www.dabeaz.com/action/showme.html
stocks = []
for line in open("portfolio.dat"):
    fields  = line.split()
    name    = fields[0]
    shares  = int(fields[1])
    price   = float(fields[2])
    holding = (name, shares, price)
    stocks.append(holding)

stocks.sort()
for s in stocks:
    print("%-10s %8d %10.2f" % s)

total = sum([s[1]*s[2] for s in stocks])
print("Total", total)
