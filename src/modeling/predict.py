def predict_user_behavior(model, scaler):
    print("\nMasukkan data untuk memprediksi kelas User Behavior:")
    app_usage_time = float(input("App Usage Time (menit per hari): "))
    screen_on_time = float(input("Screen On Time (jam per hari): "))
    battery_drain = float(input("Battery Drain (mAh per hari): "))
    num_apps_installed = int(input("Number of Apps Installed: "))
    data_usage = float(input("Data Usage (MB per hari): "))
    age = int(input("Age: "))
    operating_system = (
        1 if input("Operating System (iOS/Android): ").strip().lower() == "ios" else 0
    )
    gender = 1 if input("Gender (Male/Female): ").strip().lower() == "male" else 0

    # Mempersiapkan data input dengan urutan yang sesuai
    user_data = [
        app_usage_time,
        screen_on_time,
        battery_drain,
        num_apps_installed,
        data_usage,
        age,
        operating_system,
        gender,
    ]
    user_data_scaled = scaler.transform([user_data])  # Use the same scaler
    prediction = model.predict(user_data_scaled)
    print(
        "\nPrediksi kelas User Behavior untuk data yang Anda masukkan adalah:",
        prediction[0],
    )
