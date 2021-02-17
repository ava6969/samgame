from zeep import Client
from zeep import xsd

header = xsd.Element(
     '{http://www.xignite.com/services/}Header',
     xsd.ComplexType([
          xsd.Element(
               '{http://www.xignite.com/services/}Username',
               xsd.String()
          )
     ])
)

header_value = header(Username='387A0E3B412746A3939A43B9483B41B9')

client = Client('http://globalnews.xignite.com/xGlobalNews.asmx?WSDL')
result = client.service.ListSectors(_soapheaders=[header_value])

# A real application should include some error handling. This example just prints the response.
print(result)
