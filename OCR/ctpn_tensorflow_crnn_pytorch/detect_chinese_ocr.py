import os
import sys
import ocr
import time
import datetime
import re
import shutil
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import fengzhuang_chinese_ocr
import extract_chinese_ocr
import seg_resize as sg
from io import BytesIO
import chardet
import requests,json
from matplotlib import pyplot as plt
import hashlib

import phoenixdb
import phoenixdb.cursor

image_files = glob('./receipts/*.*')
# image_files = glob('./test_images/*.*')
out_path = './res_receipts/'
re_dir = './re/'

def upsert_ticket(conn,res_df):
	# print('start upsert')
	with conn.cursor() as curs:
	    print('start ticket')
	    for index in res_df.index:
	        try:    
	            curs.execute('upsert into "hst_app"."ticket" values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',\
	                      (res_df.loc[index]['rowkey'],res_df.loc[index]['market_id'],res_df.loc[index]['pos_id'],\
	                       res_df.loc[index]['ticket_no'],res_df.loc[index]['ticket_type'],\
	                       res_df.loc[index]['shop_id'],res_df.loc[index]['shop_name'],\
	                       res_df.loc[index]['deal_no'],res_df.loc[index]['编号'],\
	                       res_df.loc[index]['日期'],res_df.loc[index]['时间'],res_df.loc[index]['总数'],\
	                       res_df.loc[index]['应收金'],res_df.loc[index]['总金额'],\
	                       res_df.loc[index]['支付方式'],res_df.loc[index]['create_time']))
	            print('upsert ticket success')
	            # my_logger.info('upsert ticket success')
	        except:
	            print('Fail upsert ticket')
	            # my_logger.info('Fail upsert ticket')
	            curs.close()
	            # conn.close()
	    curs.close()


def upsert_ticket_item(conn,detail_df):
	# print('start upsert')
	with conn.cursor() as curs:
	    print('start ticket item')
	    for index in detail_df.index:
	        # 新加入字段
	        # ,?,?
	        # detail_df.loc[index]['ticket_rowkey'],
	        # detail_df.loc[index]['套餐'],
	        # taocan_item
	        try:
	            curs.execute('upsert into "hst_app"."ticket_item" values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',\
	                      (detail_df.loc[index]['rowkey'],detail_df.loc[index]['ticket_rowkey'],\
	                       detail_df.loc[index]['market_id'],detail_df.loc[index]['pos_id'],\
	                       detail_df.loc[index]['ticket_no'],detail_df.loc[index]['ticket_type'],\
	                       detail_df.loc[index]['shop_id'],detail_df.loc[index]['shop_name'],\
	                       detail_df.loc[index]['deal_no'],detail_df.loc[index]['编号'],detail_df.loc[index]['日期'],\
	                       detail_df.loc[index]['时间'],detail_df.loc[index]['商品'],detail_df.loc[index]['套餐'],\
	                       detail_df.loc[index]['tao_item'],\
	                       detail_df.loc[index]['数量'],detail_df.loc[index]['单价'],\
	                       detail_df.loc[index]['成交'],detail_df.loc[index]['其他'],detail_df.loc[index]['create_time']))
	            print('upsert ticket_item success')
	            # my_logger.info('upsert ticket_item success')
	        except:
	            print('Fail upsert ticket_item')
	            # my_logger.info('Fail upsert ticket_item')
	            curs.close()
	            # conn.close()
	    curs.close()


def get_re_type(image_info,box_date):
	s = image_info.split('_')[0]+'_'+image_info.split('_')[1]+'_'+image_info.split('_')[-1]
	f = open('select_version.txt','r')
	index_dict = {}
	for line in f.readlines():
	    if re.search(s,line):
	        print('find the re_type')
	        # my_logger.info('Find the re version')
	        line = line.strip()
	        index_dict = json.loads(line)
	if index_dict == {}:
	    return s+'_'
	for i,k in enumerate(index_dict[s].keys()):
	    begin = k.split('_')[0]
	    end = k.split('_')[1]
	    if box_date > begin and box_date < end:
	        f.close()
	        return s+index_dict[s][k]
	f.close()
	return s+'_'




if __name__ == '__main__':
	if os.path.exists(out_path+'images'):
	    shutil.rmtree(out_path+'images')
	os.mkdir(out_path+'images')
	if os.path.exists(out_path+'texts'):
	    shutil.rmtree(out_path+'texts')
	os.mkdir(out_path+'texts')

	database_url = 'http://dev-hadoop-3:8765/'
	conn = phoenixdb.connect(database_url, autocommit=True)

	for j, file_in in enumerate(image_files):
	    # print(file_in)
	    image_info = os.path.basename(file_in).split('.')[0]
	    print(image_info)

	    l1_dir = image_info.split('_')[0] # marketId
	    l2_dir = image_info.split('_')[1] # posId
	    ticket_no = image_info.split('_')[2]+'_'+image_info.split('_')[3]+'_'+image_info.split('_')[4]+'_'+image_info.split('_')[5]
	    ticket_type = '1'

	    image = cv2.imread(file_in)  # [:, :, ::-1]
	    proj,ratio,w = sg.get_proj_n_ratio(image)
	    #print(ratio)
	    bbox = []
	    text = []
	    seg_num, seg_list = sg.get_seg_list(proj,ratio,w)
	    for i in range(seg_num):
	        image_out = out_path+'images/'+os.path.basename(file_in).split('.')[0]+'_'+str(i)+'.png'
	        sub_img = image[seg_list[i]:seg_list[i+1],:w]
	        sub_img = sg.resize_image(sub_img)
	        result,image_framed = fengzhuang_chinese_ocr.one_img_get_text(sub_img)
	        
	        #save image
	        Image.fromarray(image_framed).save(image_out)
	        #show text
	        for key in result:
	            #print(result[key][1])
	            text.append(result[key][1])
	            #print('\n')
	        #show box
	        #for key in result:
	            #print(result[key][0])
	            #bbox.append(result[key][0])
	    #print(text)
	    #print(bbox)
	    output_txt = out_path+'texts/'+os.path.basename(file_in).split('.')[0]+'.txt'
	    with open(output_txt,'w') as f:
	        for l in text:
	            print(l)
	            line = l+'\r\n'
	            f.writelines(line)


	    # extract info and save into database
	    print(j)
	    re_type = re_dir+get_re_type(image_info+'_1','20200710000000')+'.txt'
	    res_df, detail_df, res_dict = extract_chinese_ocr.extract_info(re_type,text)
	    
	    # 根据outpath得到存放csv路径
	    res_file = out_path+'res_'+image_info.split('_')[1]+'_1'+'.csv'
	    detail_file = out_path+'detail_'+image_info.split('_')[1]+'_1'+'.csv'

	    # 当前系统时间
	    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
	    # md5(marketId+posId+ticketNo)
	    m2 = hashlib.md5()
	    s4md5 = l1_dir+l2_dir+ticket_no
	    m2.update(s4md5.encode('utf-8'))   
	    res_md5 = m2.hexdigest()

	    # 存放到对应csv

	    # res如果为空
	    if res_df.empty == True:
	        print('Empty res!')
	        # my_logger.info('Cannnot get ticket dataframe: %s'%image_info)
	        # conn.close()
	        # return
	    else:
	        #加入md5值和shop_id
	        res_df['rowkey'] = res_md5
	        res_df['market_id'] = l1_dir
	        res_df['pos_id'] = l2_dir
	        res_df['ticket_no'] = ticket_no
	        res_df['ticket_type'] = ticket_type
	        res_df['shop_id'] = '-'
	        res_df['shop_name'] = '-'
	        res_df['deal_no'] = '-'
	        res_df['create_time'] = time_str
	        res_df['日期'] = ['%s-%s-%s'% (i[0:4],i[4:6],i[6:8]) for i in res_df['日期']]
	    
	        # 打开phoenix连接
	        # print('open database')
	        # database_url = 'http://dev-hadoop-3:8765/'
	        # conn = phoenixdb.connect(database_url, autocommit=True)
	        # print('open phoenix success')

	        # print('check point 1')
	        # 写入ticket表结构
	        upsert_ticket(conn,res_df)

	    
	    #     # 根据文件名存放csv路径
	    #     res_file = root+'/res_'+image_info.split('_')[1]+'_'+image_info.split('_')[-1]+'.csv'
	    #     detail_file = root+'/detail_'+image_info.split('_')[1]+'_'+image_info.split('_')[-1]+'.csv'

	    # 放入ticket.csv
	    # res_file = root+'/ticket.csv'

	    #     # 存入汇总csv
	    if os.path.exists(res_file):
	        print('add')
	        res_df.to_csv(res_file,sep = ',',index = False,header = False,encoding='utf-8_sig',mode = 'a')
	    else:
	        print('write')
	        res_df.to_csv(res_file,sep = ',',index = False,header = True,encoding='utf-8_sig',mode = 'w')

	    # detail如果为空
	    if detail_df.empty == True:
	        print('Empty detail!')
	        # my_logger.info('Cannnot get ticket_item dataframe: %s'%image_info)
	        # conn.close()
	        # return
	    else:
	        #加入md5值
	        detail_md5 = []
	        for i in detail_df.index:
	            m2.update((s4md5+str(i)).encode('utf-8'))   
	            detail_md5.append(m2.hexdigest())
	        detail_df['rowkey'] = detail_md5
	        detail_df['ticket_rowkey'] = res_md5
	        detail_df['market_id'] = l1_dir
	        detail_df['pos_id'] = l2_dir
	        detail_df['ticket_no'] = ticket_no
	        detail_df['ticket_type'] = ticket_type
	        detail_df['shop_id'] = '-'
	        detail_df['shop_name'] = '-'
	        detail_df['deal_no'] = '-'
	        detail_df['create_time'] = time_str
	        detail_df['日期'] = ['%s-%s-%s'% (i[0:4],i[4:6],i[6:8]) for i in detail_df['日期']]
	    
	        # print('check point 2')
	        # 写入ticket_item表结构
	        upsert_ticket_item(conn,detail_df)


	        # 放入ticket_item.csv
	        # detail_file = root+'/ticket_item.csv'

	        #     # 存入明细csv
	        if os.path.exists(detail_file):
	            print('add')
	            detail_df.to_csv(detail_file,sep = ',',index = False,header = False,encoding='utf-8_sig',mode = 'a')
	        else:
	            print('write')
	            detail_df.to_csv(detail_file,sep = ',',index = False,header = True,encoding='utf-8_sig',mode = 'w')


	        # 关闭phoenix连接
	        # conn.close()

	end = time.time()
	# print('ocr task finished: %.3f seconds'%(end-begin))
	# my_logger.info('Task: %s is finished. %.3f seconds'%(image_info,(end-begin)))
	# return (end-begin)
	conn.close()
