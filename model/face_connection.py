#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:39:26 2023

@author: jkou
"""

import psycopg

class lineConnection:
    conn = None
    
# Funciones para la tablas.
    
    def __init__(self):
        try:
            self.conn = psycopg.connect("dbname=face_recognition user=esp32 password=123456 host=localhost port=5432")
        except psycopg.OperationalError as err:
            print(err)
            if self.conn is not None:
                self.conn.close()
    
    def write_entrada(self, data):
        with self.conn.cursor() as cur:
            sql = """
                INSERT INTO face_recognition (id, led_red, led_green, led_mirrow) VALUES (%(id)s, %(led_red)s, %(led_green)s, %(led_mirrow)s)
            """
            cur.execute(sql, data)
            self.conn.commit()
            
    def read_one_entrada(self, id):
        with self.conn.cursor() as cur:
            cur.execute("""
               SELECT * FROM face_recognition WHERE id = %s
            """ %(id) )
            return cur.fetchone()
        
    def read_all_entradas(self):
        with self.conn.cursor() as cur:
            
            sql = """
                SELECT * FROM face_recognition;
            """
            data = cur.execute(sql)
            return data.fetchall()
        
    def update_entrada(self, data):
         
        with self.conn.cursor() as cur:
            
            sql = """
                UPDATE face_recognition SET led_red=%(led_red)s, led_green=%(led_green)s, led_mirrow=%(led_mirrow)s WHERE id=%(id)s

            """
            cur.execute(sql,data)
            
        self.conn.commit()
    