import urllib.request, json
req = urllib.request.Request('http://localhost:8000/api/auth/login', 
    data=json.dumps({'email': 'test@example.com', 'password': 'password123'}).encode('utf-8'), 
    headers={'Content-Type': 'application/json'})
try:
    res = urllib.request.urlopen(req)
    with open('test_res.json', 'w', encoding='utf-8') as f:
        f.write(res.read().decode())
except Exception as e:
    with open('test_res.json', 'w', encoding='utf-8') as f:
        f.write(str(e))
