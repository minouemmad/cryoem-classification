import sys
import openpyxl

wb = openpyxl.load_workbook(r'docs/GCER Population Tables.xlsx', data_only=True)


def dump(sheet, maxrows=200):
    ws = wb[sheet]
    print('\n=============== SHEET:', sheet, '===============')
    rows = []
    for r in ws.iter_rows(values_only=True):
        vals = [('' if v is None else str(v).strip()) for v in r]
        if any(vals):
            while vals and vals[-1] == '':
                vals.pop()
            rows.append(vals)
    print('non-empty rows:', len(rows))
    for i, vals in enumerate(rows[:maxrows]):
        print(f'[{i}] ' + ' | '.join(vals))


if __name__ == '__main__':
    sheets = sys.argv[1:] or ['Current Processing']
    for s in sheets:
        dump(s)
