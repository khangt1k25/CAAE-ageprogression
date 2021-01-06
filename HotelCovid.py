
'''
Ý tưởng chính: 
+   Do mỗi ngày rời rạc nhau, có thể sắp giá riêng biệt; do đó có thể chia ra 
    từng ngày để tối ưu. 
+   Save_price sắp xếp là 1 con số nằm trong giá các khách hàng mong đợi. 
    ví dụ [3,6,10] thì save_price chỉ nằm trong 3 số 3,6,10 mà thôi (Chung minh duoc: chia khoang)

'''


from collections import defaultdict

if __name__ == "__main__":

    # Reading input
    with open('./input.txt','r') as f:
        lines  = f.read().splitlines()
    lines = [line.split(' ') for line in lines]
    lines = [list(map(int, line))  for line in lines]
    

    price = [ele[-1] for ele in lines]
    day_start = [ele[0]+1 for ele in lines]
    day_end = [ele[1] for ele in lines]
    point_s = min(day_start)
    point_e = max(day_end)
    
    save_price = defaultdict(int)
    total_profit = 0

    # Traverse all day possible
    for day in range(point_s, point_e+1, 1):
        # Get expected price for a day 
        price = []
        for p in range(len(lines)):
            if day>=day_start[p] and day<=day_end[p]:
                price.append(lines[p][-1])
            else:
                price.append(0)
        price = sorted(price)
        
        # Evaluate max profit can be achieved: Traverse all opt in price
        max_the_day = -1
        for opt in price:
            if opt==0:
                continue
            tmp = [ele for ele in price if ele>=opt]

            if max_the_day < len(tmp) * opt:
                max_the_day = len(tmp) * opt
                save_price[day] = opt
        
        total_profit += max_the_day
        

    print(total_profit)
    for day, price in save_price.items():
        print('{} {}'.format(day, price))
asd