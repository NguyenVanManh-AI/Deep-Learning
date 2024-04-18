### Các lưu ý

#### Tập dữ liệu 
    + Dataset.zip : tập dữ liệu gốc train , test cho file VGGFace_SENet50_Transfer_Fine_Tuning.ipynb 
    + Dataset_fine_tuning_train_valid_test.zip : Từ tập Dataset tổ chức thành dữ liệu train , test , valid để Transfer learning : Fine tuning (để nguyên ảnh)
    + Dataset_face_fine_tuning_train_valid_test.zip : Từ tập Dataset_fine_tuning_train_valid_test dùng face_recognition để trích xuất khuông mặt , resize lại thành 224*224 và lưu lại thành ảnh 

#### Ipynb
1. VGGFace_SENet50_Transfer_Feature_Extractor_Facerognition.ipynb (MAIN)
    + Với mục đích dùng cho bài toán thay đổi số lượng nhãn 
    + Sử dụng ngay tập Datatset , resize và dectect khuông mặt trên ảnh đó sau đó chuyển thành tensor 
    + Sử dụng Conv của VGG Face và FCs tự build của mình để trích xuất đặc trưng khuông mặt từ tensor ảnh 
    + Tính khoảng cách vector rồi dự đoán nhãn 
    + Tìm ngưỡng cho độ chính xác cao nhất (threshold = 15 . cho độ chính xác gần 90%) để phân biệt nhãn UNKNOW (không có trong tập vector cần tính khoảng cách)

2. VGGFace_SENet50_Transfer_Fine_Tuning.ipynb (ĐỌC THÊM)
    + Mục đích là cũng sử dụng Conv của VGG Face + FCs của mình 
    + Sau đó transfer learning : fine tuning lại bằng dữ liệu của mình để tăng độ chính xác cho nó , tuy nhiên vì dữ liệu của mình quá ít mà model lại quá phức tạp , dù đã cố gắng giảm độ phức tạp bằng cách thêm các lớp Dropdown nhưng vẫn bị overfitting 
    + Thử lưu lại model và load ra (bỏ đi lớp softmax ở cuối để dùng cho việc trích xuất vector đặc trưng) (VGGFace.TestModel_Fine tuning.Facerognition.ipynb)
    nhưng kết quả là độ chính xác chỉ khoảng 30% . 

#### Kết luận 
    + Sử dụng file VGGFace_SENet50_Transfer_Feature_Extractor_Facerognition.ipynb
    + Sử dụng tập Dataset.zip
    + Model : save_models\VGGFace_SENet_Transfer_Model.h5

#### Tham khảo & Research  
    + https://github.com/facebookresearch/faiss
    + sử dụng FAISS của facebook research để tối ưu cho việc tính khoảng cách và tự động tìm threshold
    + Tự động cập nhật threshold cho mình khi dữ liệu thay đổi (thêm/bớt vector đặc trưng - cũng như classes)
