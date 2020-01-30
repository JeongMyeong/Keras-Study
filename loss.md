## categorical_crossentropy VS sparse_categorical_crossentropy

- 만약 타겟값이 one-hot encoded 된 값이면 categorical_crossentropy 를 쓴다.  
ex) y   
　　[  
　　[1,0,0,0]  
　　[0,1,0,0]  
　　[0,0,0,1]  
　　[1,0,0,0]  
　　　　　　　]  
- 만약 타겟값이 정수형태이면 sparse_categorical_crossentropy 를 쓴다.  
ex) y    
　　[  
　　[1]  
　　[2]  
　　[3]  
　　　]  
