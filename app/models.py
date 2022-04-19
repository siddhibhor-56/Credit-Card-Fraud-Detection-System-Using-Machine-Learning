from app import db
from werkzeug.security import generate_password_hash, check_password_hash


ACCESS = {
    'User': 1,
    'Admin': 2
}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    username = db.Column(db.String(100), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    fname = db.Column(db.String(1000))
    lname = db.Column(db.String(1000))

    def __repr__(self):
        return "<User {}>".format(self.username)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.username == "Admin"
    
    # def allowed(self, access_level):
    #     return self.access >= access_level