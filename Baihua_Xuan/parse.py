import os

f = open('attributes.txt', 'r')

lines = f.readlines()

text_file = open('attr_parsed.txt', 'w')

for line in lines:
	tokens = line.split(' ')
	        
        if tokens[0] == 'VAR:' or tokens[0] == '_':
        	continue
        else:
                attr = tokens[0]
                if (attr != '' and attr[0].isalpha()):
			attr.replace(',', '')
                        attr.replace('_', '')
                        attr.replace(' ', '')
                        print attr
                	text_file.write(attr + '\n')



f.close()

text_file.close()

fil = open('attr_parsed.txt', 'r')

l = list(fil)

print l

text_file = open('attr_parsed.txt', 'w')

for element in l:
	text_file.write(element.rstrip('\n') + ' ')

text_file.close()

fil.close()
