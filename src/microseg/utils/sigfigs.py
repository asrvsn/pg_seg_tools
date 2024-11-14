import numpy as np

def format_with_err(x, std, sf, sf_std=None, **kwargs):
	if sf_std is None:
		sf_std = sf
	return f'{format(x, sf, **kwargs)} ± {format(std, sf_std, **kwargs)}'

def format(x, sf, amax: float=1e4, amin: float=1e-4):
	"""Convert a float, x, to a string with sf significant figures.

	This function returns a decimal string representation of a float
	to a specified number of significant figures.

		>>> create_string(9.80665, 3)
		'9.81'
		>>> create_string(0.0120076, 3)
		'0.0120'
		>>> create_string(100000, 5)
		'100000'

	Note the last representation is, without context, ambiguous. This
	is a good reason to use scientific notation, but it's not always
	appropriate.

	Note
	----

	Performing this operation as a set of string operations arguably
	makes more sense than a mathematical operation conceptually. It's
	the presentation of the number that is being changed here, not the
	number itself (which is in turn only approximated by a float).

	"""
	sf = int(sf)
	x = float(x)

	if sf < 1: raise ValueError("1+ significant digits required.")

	if np.isnan(x):
		return ''
	elif abs(x) > amax or abs(x) < amin: 
		return ''.join(('{:.', str(sf - 1), 'e}')).format(x)
	else:	
		# retrieve the significand and exponent from the S.N. form
		s, e = ''.join(( '{:.', str(sf - 1), 'e}')).format(x).split('e')
		e = int(e) # might as well coerce now

		if e == 0:
			# Significand requires no adjustment
			return s

		s = s.replace('.', '')
		if e < 0:
			# Placeholder zeros need creating 
			return ''.join(('0.', '0' * (abs(e) - 1), s))
		else:
			# Decimal place need shifting
			s += '0' * (e - sf + 1) # s now has correct s.f.
			i = e + 1
			sep = ''
			if i < sf: sep = '.'
			if s[0] == '-': i += 1
			return sep.join((s[:i], s[i:]))

