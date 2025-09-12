import profilePic from "./assets/profilepic.jpg";

function Card() {
  return (
    <div className="card">
      <img src={profilePic} alt="Profile Picture" className="img-profile" />
      <h2 className="card-title">Justine Romales</h2>
      <p>I am a Student</p>
    </div>
  );
}

export default Card;
